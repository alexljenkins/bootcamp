
from queue import PriorityQueue

class BaseState(object):
    def __init__(self, value, parent,
                    start = 0, goal = 0):
        self.children = []
        self.parent = parent
        self.value = value
        self.distance = 0

        if parent:
            self.path = parent.path[:] # creates copy instead of pointer
            self.path.append(value)
            self.start = parent.path
            self.goal = parent.goal

        else:
            self.path = [value]
            self.start = start
            self.goal = goal

    def get_distance(self):
        pass
    def create_children(self):
        pass

class StateString(BaseState):
    def __init__(self, value, parent,
                    start = 0, goal = 0):
        super(StateString, self).__init__(value, parent, start, goal)
        self.distance = self.get_distance()

    def get_distance(self):
        if self.value == self.goal:
            return 0 # we've reached the goal
        dist = 0
        for i in range(len(self.goal)):
            letter = self.goal[i]
            dist += abs(i-self.value.index(letter))

        return dist

    def create_children(self):
        if not self.children:
            for i in range(len(self.goal)-1):
                val = self.value
                val = val[:i] + val[i+1] + val[i] + val[i+2:]
                child = StateString(val, self) # pass self to store parent as well
                self.children.append(child)


class AStarSolver:
    def __init__(self, start, goal):
        self.path = []
        self.visited = []
        self.priority_q = PriorityQueue()
        self.start = start
        self.goal = goal

    def find_path(self):
        start_state = StateString(self.start, 0, self.start, self.goal)
        count = 0 #counts the created childs

        # "put" adds a tuple to the priority queue method
        self.priority_q.put((0, count, start_state)) # priority, children, states

        while(not self.path and self.priority_q.qsize()): # while priority_q has values still in it
            # get the highest priorty item in que and grab the "start_state" value
            closest_child = self.priority_q.get()[2]
            closest_child.create_children()
            self.visited.append(closest_child.value)

            for child in closest_child.children:
                if child.value not in self.visited:
                    count += 1
                    if not child.distance: # if child.distance doesn't exist, solved!
                        self.path = child.path #2. breaks the while loop
                        break #1. breaks the for loop
                    self.priority_q.put((child.distance, count, child))

        if not self.path:
            print(f"Goal of {self.goal} is not possible")

        return self.path


if __name__ == "__main__":
    start1 = "xlea"
    goal1 = "alex"
    a = AStarSolver(start1, goal1)
    a.find_path()
    print(a.path)
