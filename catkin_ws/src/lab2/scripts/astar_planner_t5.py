#!/usr/bin/env python3

import math
import heapq
import rospy

from nav_msgs.msg import OccupancyGrid, GridCells, Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Point

"""
Professor-style global containers.

These mirror the original A* structure:
path        = ordered list of cells from start to goal
expanded    = dictionary of expanded cells
frontier    = dictionary of discovered/frontier cells
came_from   = parent map for path reconstruction
cost_so_far = g-cost map
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

        self.occupied_threshold = rospy.get_param("~occupied_threshold", 50)
        self.allow_unknown = rospy.get_param("~allow_unknown", False)
        self.heuristic_type = rospy.get_param("~heuristic", "manhattan")

        # Subscribers
        self.map_sub = rospy.Subscriber("/map", OccupancyGrid, self.map_callback)
        self.start_sub = rospy.Subscriber("/initialpose", PoseWithCovarianceStamped, self.initialpose_callback)
        self.goal_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.goal_callback)

        # Task 4 publishers: GridCells and raw A* path
        self.expanded_pub = rospy.Publisher("/astar/expanded", GridCells, queue_size=1, latch=True)
        self.frontier_pub = rospy.Publisher("/astar/frontier", GridCells, queue_size=1, latch=True)
        self.unexplored_pub = rospy.Publisher("/astar/unexplored", GridCells, queue_size=1, latch=True)
        self.path_cells_pub = rospy.Publisher("/astar/path_cells", GridCells, queue_size=1, latch=True)
        self.path_pub = rospy.Publisher("/astar/path", Path, queue_size=1, latch=True)

        # Task 5 publisher: optimized waypoint path
        self.waypoints_pub = rospy.Publisher("/astar/waypoints", Path, queue_size=1, latch=True)

        rospy.loginfo("A* ROS node started.")
        rospy.loginfo("Waiting for /map, /initialpose, and /move_base_simple/goal.")

    def map_callback(self, msg):
        self.map_msg = msg
        self.width = msg.info.width
        self.height = msg.info.height
        self.resolution = msg.info.resolution
        self.origin_x = msg.info.origin.position.x
        self.origin_y = msg.info.origin.position.y

        rospy.loginfo_once(
            "Map received: width=%d height=%d resolution=%.3f frame=%s",
            self.width,
            self.height,
            self.resolution,
            msg.header.frame_id
        )

    def initialpose_callback(self, msg):
        global start

        wx = msg.pose.pose.position.x
        wy = msg.pose.pose.position.y

        start = self.world_to_grid(wx, wy)

        if start is None:
            rospy.logerr("Start pose is outside the map.")
            return

        rospy.loginfo("Start set from RViz 2D Pose Estimate: %s", start)
        self.try_search()

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
            rospy.logerr("Start cell is occupied or unknown: %s", start)
            return

        if not self.is_free(end):
            rospy.logerr("Goal cell is occupied or unknown: %s", end)
            return

        rospy.loginfo("Running A* from %s to %s", start, end)

        self.search()

        unexplored = self.compute_unexplored(max_cells=4000)

        # Task 4 outputs
        self.publish_gridcells(self.expanded_pub, expanded.keys())
        self.publish_gridcells(self.frontier_pub, frontier.keys())
        self.publish_gridcells(self.unexplored_pub, unexplored)
        self.publish_gridcells(self.path_cells_pub, path)
        self.publish_path(path)

        # Task 5 output
        waypoints = self.reduce_waypoints(path)
        self.publish_waypoints(waypoints)

        rospy.loginfo("Expanded nodes: %d", len(expanded))

        if end in cost_so_far:
            rospy.loginfo("Final cost: %.3f", cost_so_far[end])
            rospy.loginfo("Raw A* path length: %d", len(path))
            rospy.loginfo("Optimized waypoint length: %d", len(waypoints))
        else:
            rospy.logwarn("No path found.")

    def search(self):
        """
        Professor-style A* search adapted from the image-map structure.
        """
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

                new_cost = cost_so_far[current] + 1.0

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
        if self.heuristic_type == "euclidean":
            return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

        # Default: Manhattan, matching the professor's 4-connected A* structure.
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
        idx = self.grid_to_index(x, y)
        value = self.map_msg.data[idx]

        if value == -1:
            return self.allow_unknown

        if value >= self.occupied_threshold:
            return False

        return True

    def compute_unexplored(self, max_cells=4000):
        """
        For RViz display only.
        Publishes a limited number of free cells that were not expanded,
        not frontier, and not path. Limiting keeps RViz responsive.
        """
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

    def reduce_waypoints(self, full_path):
        """
        Task 5:
        Reduce the raw A* cell-by-cell path into fewer waypoints.

        Example:
        [(1,2), (2,2), (3,2), (4,2), (5,2)]
        becomes:
        [(1,2), (5,2)]

        The algorithm keeps:
        - the first point
        - any point where the path direction changes
        - the final point
        """
        if len(full_path) <= 2:
            return list(full_path)

        reduced = [full_path[0]]

        prev_dx = full_path[1][0] - full_path[0][0]
        prev_dy = full_path[1][1] - full_path[0][1]

        for i in range(2, len(full_path)):
            curr_dx = full_path[i][0] - full_path[i - 1][0]
            curr_dy = full_path[i][1] - full_path[i - 1][1]

            if (curr_dx, curr_dy) != (prev_dx, prev_dy):
                reduced.append(full_path[i - 1])

            prev_dx = curr_dx
            prev_dy = curr_dy

        reduced.append(full_path[-1])
        return reduced

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

    def publish_waypoints(self, waypoint_cells):
        """
        Publish the optimized waypoint path as nav_msgs/Path.
        This is the Task 5 output topic: /astar/waypoints.
        """
        msg = Path()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.map_msg.header.frame_id

        for gx, gy in waypoint_cells:
            wx, wy = self.grid_to_world(gx, gy)

            pose = PoseStamped()
            pose.header.stamp = msg.header.stamp
            pose.header.frame_id = msg.header.frame_id
            pose.pose.position.x = wx
            pose.pose.position.y = wy
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0

            msg.poses.append(pose)

        self.waypoints_pub.publish(msg)
        rospy.loginfo("Published optimized waypoints: %d poses to /astar/waypoints", len(msg.poses))


if __name__ == "__main__":
    try:
        node = AStarPlannerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass