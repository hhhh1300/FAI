# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018
# Modified by Shang-Tse Chen (stchen@csie.ntu.edu.tw) on 03/03/2022

"""
This is the main entry point for HW1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)

import queue
import sys
def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "astar": astar,
        "astar_corner": astar_corner,
        "astar_multi": astar_multi,
        "fast": fast,
    }.get(searchMethod)(maze)

def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    start_r, start_c = maze.getStart()
    objs = maze.getObjectives()
    dim_r, dim_c = maze.getDimensions()
    visited = [[False for _ in range(dim_c)] for _ in range(dim_r)]
    # print(len(visited), len(visited[0]))
    q = queue.Queue()
    path = [(start_r, start_c)]
    visited[start_r][start_c] = True
    q.put((start_r, start_c, path))
    ans = []
    while not q.empty():
        now_r, now_c, now_path = q.get()
        # print(now_r, now_c)
        isFind = True
        for r, c in objs:
            if now_r != r or now_c != c or not maze.isValidPath(now_path):
                isFind = False
                break
        if isFind:
            ans = now_path
            break
        for r, c in maze.getNeighbors(now_r, now_c):
            if (not visited[r][c]) and maze.isValidMove(r, c):
                visited[r][c] = True
                q.put((r, c, now_path + [(r, c)]))

    if maze.isValidPath(ans):
        return ans
    return []


def astar(maze):
    """
    Runs A star for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    start_r, start_c = maze.getStart()
    objs = maze.getObjectives()
    def h(r, c):
        return abs(r-objs[0][0]) + abs(c-objs[0][1])
    dim_r, dim_c = maze.getDimensions()
    min_cost = [[sys.maxsize for _ in range(dim_c)] for _ in range(dim_r)]
    visited = [[False for _ in range(dim_c)] for _ in range(dim_r)]
    # print(len(visited), len(visited[0]))
    q = queue.PriorityQueue()
    path = [(start_r, start_c)]
    min_cost[start_r][start_c] = h(start_r, start_c)
    q.put((0+h(start_r, start_c), 0, path))
    ans = []
    while not q.empty():
        _ , cost, now_path = q.get()
        now_r = now_path[-1][0]
        now_c = now_path[-1][1]
        isFind = True
        for r, c in objs:
            if now_r != r or now_c != c or not maze.isValidPath(now_path):
                isFind = False
                break
        if isFind:
            ans = now_path
            break
        # visited[now_r][now_c] = True
        for r, c in maze.getNeighbors(now_r, now_c):
            f = cost+1+h(r, c)
            if min_cost[r][c] > f and maze.isValidMove(r, c):
                min_cost[r][c] = f
                q.put((f, cost+1, now_path + [(r, c)]))

    if maze.isValidPath(ans):
        return ans
    return []

import copy
def getRealDistance(maze, From, To, info):
    if (From, To) in info:
        return info[(From, To)]
    start_temp = maze.getStart()
    objs_temp = maze.getObjectives()
    maze.setObjectives([To])
    maze.setStart(From)
    maze2 = copy.deepcopy(maze)
    info[(From, To)] = len(bfs(maze2))
    maze.setObjectives(objs_temp)
    maze.setStart(start_temp)
    return info[(From, To)]

def astar_corner(maze):
    """
    Runs A star for part 2 of the assignment in the case where there are four corner objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
        """
    start_r, start_c = maze.getStart()
    objs = maze.getObjectives()
    # def h(r, c, now_Found):
    #     cost = 0
    #     for obj_r, obj_c in objs:
    #         if not (obj_r, obj_c) in now_Found:
    #             cost = max(cost, abs(r - obj_r) + abs(c - obj_c))
    #     return cost
    def h(r, c, now_Found):
        cost = 0
        # corners = [x for x in objs if x not in now_Found]
        corners = [objs[id] for id in range(len(objs)) if not now_Found[id]]
        # print(corners)
        while len(corners):
            closest_corner, curr_min_cost = tuple(), sys.maxsize
            for obj_r, obj_c in corners:
                now_cost = abs(r - obj_r) + abs(c - obj_c)
                if now_cost < curr_min_cost:
                    curr_min_cost = now_cost
                    closest_corner = (obj_r, obj_c)
            corners.remove(closest_corner)
            r, c = closest_corner
            cost += curr_min_cost
        return cost
    min_cost = {}
    visited = set()
    q = queue.PriorityQueue()
    path = [(start_r, start_c)]
    has_Found = tuple(False for _ in range(len(objs)))
    min_cost[(start_r, start_c, has_Found)] = h(start_r, start_c, has_Found)
    q.put((h(start_r, start_c, has_Found), 0, path, has_Found))
    ans = []
    while not q.empty():
        _ , cost, now_path, now_Found = q.get()
        now_r = now_path[-1][0]
        now_c = now_path[-1][1]
        isFind = True
        for id in range(len(objs)):
            if now_r == objs[id][0] and now_c == objs[id][1]:
                temp = list(now_Found)
                temp[id] = True
                now_Found = tuple(temp)
                break
        for id in range(len(objs)):
            if not now_Found[id] or not maze.isValidPath(now_path):
                isFind = False
                break
        # print(now_path, now_Found)
        if isFind:
            ans = now_path
            break
        if (now_r, now_c, now_Found) not in visited :
            visited.add((now_r, now_c, now_Found))
            for r, c in maze.getNeighbors(now_r, now_c):
                f = cost + 1 + h(r, c, now_Found)
                if (r, c, now_Found) in min_cost:
                    if f > min_cost[r, c, now_Found] + h(r, c, now_Found):
                        continue
                min_cost[r, c, now_Found] = f
                if maze.isValidMove(r, c):
                    q.put((f, cost+1, now_path + [(r, c)], now_Found))

    if maze.isValidPath(ans):
        return ans
    return []

def astar_multi(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    def find(parent, node):
        if parent[node] == node:
            return node
        node = find(parent, parent[node])
        return node
    def union(parent, x, y):
        parent_x = find(parent, x)
        parent_y = find(parent, y)
        if parent_x == parent_y:
            return
        parent[parent_x] = parent_y
    
    start_r, start_c = maze.getStart()
    objs = maze.getObjectives()
    info = {}
    info2 = {}
    info_mst = {}
    # def h(r, c, now_Found):
    #     cost = 0
    #     unvisited_objs = [objs[id] for id in range(len(objs)) if not now_Found[id]]
    #     if (r, c, now_Found) in info_mst:
    #         return info_mst[r, c, now_Found] + min([abs(obj[0] - r) + abs(obj[1] - c) for obj in unvisited_objs])
    #     # unvisited_objs = [x for x in objs if x not in now_Found]
    #     edges = []
    #     for obj1 in range(len(unvisited_objs)):
    #         for obj2 in range(len(unvisited_objs)):
    #             if obj1 != obj2:
    #                 edges.append((obj1, obj2, abs(unvisited_objs[obj1][0] - unvisited_objs[obj2][0]) + abs(unvisited_objs[obj1][1] - unvisited_objs[obj2][1])))
    #     edges = sorted(edges, key=lambda edge : edge[2])
    #     parent = [x for x in range(len(unvisited_objs))]
    #     for obj1, obj2, distance in edges:
    #         if find(parent, obj1) != find(parent, obj2):
    #             union(parent, obj1, obj2)
    #             cost += distance
    #     info_mst[r, c, now_Found] = cost
    #     return cost + min([abs(obj[0] - r) + abs(obj[1] - c) for obj in unvisited_objs])
    def h(r, c, now_Found):
        cost = 0
        unvisited_objs = [objs[id] for id in range(len(objs)) if not now_Found[id]]
        visited_objs = []
        if (r, c, now_Found) in info_mst:
            return info_mst[(r, c, now_Found)] + min([abs(obj[0] - r) + abs(obj[1] - c) for obj in unvisited_objs])
        nodes = [[to for to in range(len(unvisited_objs)) if fr != to] for fr in range(len(unvisited_objs))]
        visited1 = set()
        visited2 = set([id for id in range(len(objs)) if not now_Found[id]])
        pq = queue.PriorityQueue()
        pq.put((0, 0))
        while len(visited2) > 0:
            c, node = pq.get()
            if node in visited1:
                continue        
            visited2.remove(node)
            visited1.add(node)
            cost += c
            for to in (visited2):
                if to not in visited1:
                    # print(node, to)
                    pq.put((abs(unvisited_objs[node][0] - unvisited_objs[to][0]) + abs(unvisited_objs[node][1] - unvisited_objs[to][1]), to))
        info_mst[(r, c, now_Found)] = cost
        return cost + min([abs(obj[0] - r) + abs(obj[1] - c) for obj in unvisited_objs])

    # def h(r, c, now_Found):
    #     cost = sys.maxsize
    #     unvisited_objs = [objs[id] for id in range(len(objs)) if not now_Found[id]]
    #     for obj_r, obj_c in unvisited_objs:
    #         cost = max(cost,  abs(obj_r - r) + abs(obj_c - c))
    #     return cost
    # def h(r, c, now_Found):
    #     cost = sys.maxsize
    #     distance = 0
    #     unvisited_objs = [objs[id] for id in range(len(objs)) if not now_Found[id]]
    #     if now_Found in info2:
    #         for obj1 in unvisited_objs:
    #             cost = min(cost, getRealDistance(maze, (r, c), obj1, info))
    #         return cost + info2[now_Found]
    #     for obj1 in unvisited_objs:
    #         cost = min(cost, getRealDistance(maze, (r, c), obj1, info))
    #         for obj2 in unvisited_objs:
    #             if obj2 == obj1:
    #                 continue
    #             distance = max(distance, getRealDistance(maze, obj1, obj2, info))
    #     info2[now_Found] = distance
    #     return cost + distance if len(unvisited_objs) else distance
    min_cost = {}
    visited = set()
    q = queue.PriorityQueue()
    path = [(start_r, start_c)]
    has_Found = tuple(False for _ in range(len(objs)))
    min_cost[(start_r, start_c, has_Found)] = h(start_r, start_c, has_Found)
    q.put((h(start_r, start_c, has_Found), 0, path, has_Found))
    ans = []
    while not q.empty():
        _ , cost, now_path, now_Found = q.get()
        now_r = now_path[-1][0]
        now_c = now_path[-1][1]
        isFind = True
        for id in range(len(objs)):
            if now_r == objs[id][0] and now_c == objs[id][1]:
                temp = list(now_Found)
                temp[id] = True
                now_Found = tuple(temp)
                break
        for id in range(len(objs)):
            if not now_Found[id]:
                isFind = False
                break
        # print(now_path, now_Found)
        if isFind:
            ans = now_path
            break
        if (now_r, now_c, now_Found) not in visited :
            visited.add((now_r, now_c, now_Found))
            for r, c in maze.getNeighbors(now_r, now_c):
                # print(h(r, c, now_Found))
                f = cost + 1 + h(r, c, now_Found)
                if (r, c, now_Found) in min_cost:
                    if f > min_cost[r, c, now_Found] + h(r, c, now_Found):
                        continue
                min_cost[r, c, now_Found] = f
                if maze.isValidMove(r, c):
                    q.put((f, cost+1, now_path + [(r, c)], now_Found))
    # print(ans)
    if maze.isValidPath(ans):
        return ans
    return []


def fast(maze):
    """
    Runs suboptimal search algorithm for part 4.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    start_r, start_c = maze.getStart()
    r, c = start_r, start_c
    objs = maze.getObjectives()
    ans = [(r, c)]
    has_Found = tuple(False for _ in range(len(objs)))
    def h(r, c, has_Found):
        min_id, cost = 0, sys.maxsize
        for id in range(len(objs)):
            if not has_Found[id] and abs(objs[id][0] - r) + abs(objs[id][1] - c) < cost:
                min_id, cost = id, abs(objs[id][0] - r) + abs(objs[id][1] - c)
        return min_id
    while not all(has_Found):
        # print("start from :", r, c)
        target_id = h(r, c, has_Found)
        target_r, target_c = objs[target_id]
        q = queue.Queue()
        visited = set()
        visited.add((r, c))
        q.put((r, c, []))
        # print("target :", target_r, target_c)
        while not q.empty():
            now_r, now_c, now_path = q.get()
            if now_r == target_r and now_c == target_c:
                temp = list(has_Found)
                temp[target_id] = True
                has_Found = tuple(temp)
                # print(now_path)
                ans += now_path
                break
            for obj_r, obj_c in maze.getNeighbors(now_r, now_c):
                if (obj_r, obj_c) not in visited and maze.isValidMove(obj_r, obj_c):
                    q.put((obj_r, obj_c, now_path + [(obj_r, obj_c)]))
                    visited.add((obj_r, obj_c))
        r, c = target_r, target_c
    # print(ans)
    if maze.isValidPath(ans):
        return ans
    return []
