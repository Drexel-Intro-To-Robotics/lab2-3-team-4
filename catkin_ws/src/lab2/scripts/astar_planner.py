#!/usr/bin/env python3

import math
import heapq
import rospy

from nav_msgs.msg import OccupancyGrid, GridCells, Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Point


"""
Normal A* planner node for Tasks 3 and 4.

This version includes footprint-aware planning:
- Computes a clearance grid from obstacles.
- Blocks cells that are too close to obstacles.
- Adds extra cost to cells near obstacles.
- Publishes the raw A* path on /astar/path.
"""

start = None
end = None

path = []
expanded = {}
frontier = {}

came_from = {}
cost_so_far = {}


class AStarPlannerNode(object):
    def __init__(self):
        rospy.init_node("astar_planner_node")

        self.map_msg = None
        self.width = 0
        self.height = 0
        self.resolution = 0.0
        self.origin_x = 0.0
        self.origin_y = 0.0

        # Map / A* settings
        self.occupied_threshold = rospy.get_param("~occupied_threshold", 50)
        self.allow_unknown = rospy.get_param("~allow_unknown", False)
        self.heuristic_type = rospy.get_param("~heuristic", "manhattan")

        # Footprint-aware planning settings
        self.use_footprint_cost = rospy.get_param("~use_footprint_cost", True)

        self.robot_width_m = rospy.get_param("~robot_width_m", 0.28)
        self.robot_length_m = rospy.get_param("~robot_length_m", 0.30)
        self.safety_margin_m = rospy.get_param("~safety_margin_m", 0.02)

        self.robot_radius_m = 0.5 * max(self.robot_width_m, self.robot_length_m)
        self.preferred_clearance_m = self.robot_radius_m + self.safety_margin_m

        # Tune this based on your map.
        # 0.05 is one grid cell if map resolution is 0.05 m/cell.
        self.hard_clearance_m = rospy.get_param("~hard_clearance_m", 0.05)
        self.footprint_penalty = rospy.get_param("~footprint_penalty", 8.0)

        # Optional old soft-buffer system.
        # Leave false when using footprint-aware clearance.
        self.use_soft_buffer = rospy.get_param("~use_soft_buffer", False)
        self.obstacle_buffer_cells = rospy.get_param("~obstacle_buffer_cells", 1)
        self.obstacle_buffer_penalty = rospy.get_param("~obstacle_buffer_penalty", 2.0)

        self.clearance_grid = None

        # Subscribers
        self.map_sub = rospy.Subscriber(
            "/map",
            OccupancyGrid,
            self.map_callback
        )

        self.start_sub = rospy.Subscriber(
            "/initialpose",
            PoseWithCovarianceStamped,
            self.initialpose_callback
        )

        self.goal_sub = rospy.Subscriber(
            "/move_base_simple/goal",
            PoseStamped,
            self.goal_callback
        )

        # Publishers
        self.expanded_pub = rospy.Publisher(
            "/astar/expanded",
            GridCells,
            queue_size=1,
            latch=True
        )

        self.frontier_pub = rospy.Publisher(
            "/astar/frontier",
            GridCells,
            queue_size=1,
            latch=True
        )

        self.unexplored_pub = rospy.Publisher(
            "/astar/unexplored",
            GridCells,
            queue_size=1,
            latch=True
        )

        self.path_cells_pub = rospy.Publisher(
            "/astar/path_cells",
            GridCells,
            queue_size=1,
            latch=True
        )

        self.path_pub = rospy.Publisher(
            "/astar/path",
            Path,
            queue_size=1,
            latch=True
        )

        rospy.loginfo("Footprint-aware A* ROS node started.")
        rospy.loginfo("Waiting for /map, /initialpose, and /move_base_simple/goal.")

    def map_callback(self, msg):
        self.map_msg = msg
        self.width = msg.info.width
        self.height = msg.info.height
        self.resolution = msg.info.resolution
        self.origin_x = msg.info.origin.position.x
        self.origin_y = msg.info.origin.position.y

        self.compute_clearance_grid()

        rospy.loginfo_once(
            "Map received: width=%d height=%d resolution=%.3f frame=%s",
            self.width,
            self.height,
            self.resolution,
            msg.header.frame_id
        )

        rospy.loginfo_once(
            "Footprint settings: robot_radius=%.3f preferred_clearance=%.3f hard_clearance=%.3f",
            self.robot_radius_m,
            self.preferred_clearance_m,
            self.hard_clearance_m
        )

    def initialpose_callback(self, msg):
        global start, end

        wx = msg.pose.pose.position.x
        wy = msg.pose.pose.position.y

        start = self.world_to_grid(wx, wy)

        if start is None:
            rospy.logerr("Start pose is outside the map.")
            return

        rospy.loginfo("Start set from RViz 2D Pose Estimate: %s", start)

        self.clear_visualisation()

        end = None

        rospy.logwarn("Old goal cleared. Use RViz 2D Nav Goal to set a new goal.")

    def goal_callback(self, msg):
        global end

        wx = msg.pose.position.x
        wy = msg.pose.position.y

        end = self.world_to_grid(wx, wy)

        if end is None:
            rospy.logerr("Goal pose is outside the map.")
            return

        rospy.loginfo("Goal set from RViz 2D Nav Goal: %s", end)

        self.try_search()

    def try_search(self):
        global start, end

        if self.map_msg is None:
            rospy.logwarn("No /map received yet.")
            return

        if start is None:
            rospy.logwarn("No start set yet. Use RViz 2D Pose Estimate.")
            return

        if end is None:
            rospy.logwarn("No goal set yet. Use RViz 2D Nav Goal.")
            return

        if not self.is_free(start):
            rospy.logerr("Start cell is occupied, unknown, or too close to obstacle: %s", start)
            return

        if not self.is_free(end):
            rospy.logerr("Goal cell is occupied, unknown, or too close to obstacle: %s", end)
            return

        rospy.loginfo("Running footprint-aware A* from %s to %s", start, end)

        self.search()

        unexplored = self.compute_unexplored(max_cells=4000)

        self.publish_gridcells(self.expanded_pub, expanded.keys())
        self.publish_gridcells(self.frontier_pub, frontier.keys())
        self.publish_gridcells(self.unexplored_pub, unexplored)
        self.publish_gridcells(self.path_cells_pub, path)
        self.publish_path(path)

        rospy.loginfo("Expanded nodes: %d", len(expanded))

        if end in cost_so_far:
            rospy.loginfo("Final cost: %.3f", cost_so_far[end])
            rospy.loginfo("Path length: %d", len(path))
        else:
            rospy.logwarn("No path found.")

    def search(self):
        global path, expanded, frontier, came_from, cost_so_far, start, end

        path[:] = []
        expanded.clear()
        frontier.clear()
        came_from.clear()
        cost_so_far.clear()

        pq = []
        heapq.heappush(pq, (0.0, start))

        came_from[start] = None
        cost_so_far[start] = 0.0
        frontier[start] = True

        while pq and not rospy.is_shutdown():
            current_priority, current = heapq.heappop(pq)

            if current in expanded:
                continue

            expanded[current] = True

            if current in frontier:
                del frontier[current]

            if current == end:
                break

            for neighbor in self.neighbors_4(current):
                if not self.is_free(neighbor):
                    continue

                step_cost = 1.0
                clearance_cost = self.clearance_penalty(neighbor)
                soft_buffer_cost = self.obstacle_penalty(neighbor)

                new_cost = (
                    cost_so_far[current]
                    + step_cost
                    + clearance_cost
                    + soft_buffer_cost
                )

                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    came_from[neighbor] = current
                    frontier[neighbor] = True

                    h = self.heuristic(neighbor, end)
                    priority = new_cost + h
                    heapq.heappush(pq, (priority, neighbor))

        if end in came_from:
            node = end

            while node is not None:
                path.append(node)
                node = came_from[node]

            path.reverse()

    def heuristic(self, a, b):
        if self.heuristic_type == "dijkstra":
            return 0.0

        if self.heuristic_type == "euclidean":
            return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def neighbors_4(self, current):
        x, y = current

        candidates = [
            (x + 1, y),
            (x - 1, y),
            (x, y + 1),
            (x, y - 1)
        ]

        valid = []

        for nx, ny in candidates:
            if nx < 0 or nx >= self.width:
                continue

            if ny < 0 or ny >= self.height:
                continue

            valid.append((nx, ny))

        return valid

    def is_free(self, cell):
        x, y = cell

        if x < 0 or x >= self.width:
            return False

        if y < 0 or y >= self.height:
            return False

        idx = self.grid_to_index(x, y)
        value = self.map_msg.data[idx]

        if value == -1:
            return self.allow_unknown

        if value >= self.occupied_threshold:
            return False

        if self.use_footprint_cost and self.clearance_grid is not None:
            clearance = self.clearance_grid[idx]

            if clearance < self.hard_clearance_m:
                return False

        return True

    def compute_clearance_grid(self):
        """
        Computes distance from every cell to the nearest occupied cell.

        This gives A* a sense of robot footprint clearance.
        """

        if self.map_msg is None:
            return

        self.clearance_grid = [float("inf")] * (self.width * self.height)

        obstacle_cells = []

        for y in range(self.height):
            for x in range(self.width):
                idx = self.grid_to_index(x, y)
                value = self.map_msg.data[idx]

                if value >= self.occupied_threshold:
                    self.clearance_grid[idx] = 0.0
                    obstacle_cells.append((x, y))

        queue = list(obstacle_cells)
        head = 0

        neighbors = [
            (1, 0, 1.0),
            (-1, 0, 1.0),
            (0, 1, 1.0),
            (0, -1, 1.0),
            (1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)),
            (-1, -1, math.sqrt(2.0))
        ]

        while head < len(queue):
            x, y = queue[head]
            head += 1

            current_idx = self.grid_to_index(x, y)
            current_distance = self.clearance_grid[current_idx]

            for dx, dy, step in neighbors:
                nx = x + dx
                ny = y + dy

                if nx < 0 or nx >= self.width:
                    continue

                if ny < 0 or ny >= self.height:
                    continue

                nidx = self.grid_to_index(nx, ny)
                new_distance = current_distance + step * self.resolution

                if new_distance < self.clearance_grid[nidx]:
                    self.clearance_grid[nidx] = new_distance
                    queue.append((nx, ny))

        rospy.loginfo("Clearance grid computed using %d obstacle cells.", len(obstacle_cells))

    def clearance_penalty(self, cell):
        """
        Adds extra cost when the robot centre is close to obstacles.
        """

        if not self.use_footprint_cost:
            return 0.0

        if self.clearance_grid is None:
            return 0.0

        x, y = cell
        idx = self.grid_to_index(x, y)

        clearance = self.clearance_grid[idx]

        if clearance >= self.preferred_clearance_m:
            return 0.0

        clearance_error = self.preferred_clearance_m - clearance
        normalised_error = clearance_error / max(self.preferred_clearance_m, 0.001)

        return self.footprint_penalty * normalised_error

    def obstacle_penalty(self, cell):
        """
        Optional old one-cell soft buffer.

        This is off by default because footprint-aware clearance is smoother.
        """

        if not self.use_soft_buffer:
            return 0.0

        x, y = cell

        for dx in range(-self.obstacle_buffer_cells, self.obstacle_buffer_cells + 1):
            for dy in range(-self.obstacle_buffer_cells, self.obstacle_buffer_cells + 1):
                nx = x + dx
                ny = y + dy

                if nx < 0 or nx >= self.width:
                    continue

                if ny < 0 or ny >= self.height:
                    continue

                idx = self.grid_to_index(nx, ny)
                value = self.map_msg.data[idx]

                if value >= self.occupied_threshold:
                    distance = math.sqrt(dx * dx + dy * dy)

                    if distance == 0:
                        return self.obstacle_buffer_penalty

                    return self.obstacle_buffer_penalty / distance

        return 0.0

    def compute_unexplored(self, max_cells=4000):
        path_set = set(path)
        unexplored = []

        for y in range(self.height):
            for x in range(self.width):
                cell = (x, y)

                if cell in expanded:
                    continue

                if cell in frontier:
                    continue

                if cell in path_set:
                    continue

                if self.is_free(cell):
                    unexplored.append(cell)

                if len(unexplored) >= max_cells:
                    return unexplored

        return unexplored

    def world_to_grid(self, wx, wy):
        if self.map_msg is None:
            return None

        gx = int((wx - self.origin_x) / self.resolution)
        gy = int((wy - self.origin_y) / self.resolution)

        if gx < 0 or gx >= self.width:
            return None

        if gy < 0 or gy >= self.height:
            return None

        return (gx, gy)

    def grid_to_world(self, gx, gy):
        wx = self.origin_x + (gx + 0.5) * self.resolution
        wy = self.origin_y + (gy + 0.5) * self.resolution

        return wx, wy

    def grid_to_index(self, gx, gy):
        return gy * self.width + gx

    def publish_gridcells(self, publisher, cells):
        msg = GridCells()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.map_msg.header.frame_id

        msg.cell_width = self.resolution
        msg.cell_height = self.resolution

        for gx, gy in cells:
            wx, wy = self.grid_to_world(gx, gy)

            p = Point()
            p.x = wx
            p.y = wy
            p.z = 0.0

            msg.cells.append(p)

        publisher.publish(msg)

    def publish_path(self, path_cells):
        msg = Path()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.map_msg.header.frame_id

        for gx, gy in path_cells:
            wx, wy = self.grid_to_world(gx, gy)

            pose = PoseStamped()
            pose.header.stamp = msg.header.stamp
            pose.header.frame_id = msg.header.frame_id
            pose.pose.position.x = wx
            pose.pose.position.y = wy
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0

            msg.poses.append(pose)

        self.path_pub.publish(msg)

    def clear_visualisation(self):
        if self.map_msg is None:
            return

        empty_path = Path()
        empty_path.header.stamp = rospy.Time.now()
        empty_path.header.frame_id = self.map_msg.header.frame_id
        empty_path.poses = []

        self.path_pub.publish(empty_path)

        rospy.loginfo("Cleared old path.")


if __name__ == "__main__":
    try:
        node = AStarPlannerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass