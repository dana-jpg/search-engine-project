from copy import deepcopy
from queue import Queue


class PriorityQueue:
    def __init__(self, num_priorities : int = 4):
        """
        Creates an object of a PriorityQueue instance. Creates a queue for each Priority level.
        Priority levels range [1, num_priorities].
        :param num_priorities: The number of different priority levels.
        """
        self.prio_dict = {}
        for prio in range(1, num_priorities + 1):
            self.prio_dict[prio] = Queue()

    def empty(self) -> bool:
        """
        Tells if the PriorityQueue is empty. (If the lists in all priorities are empty)
        :return: Boolean indictating whether the PriorityQueue contains elements.
        """
        for prio in self.prio_dict:
            if not self.prio_dict[prio].empty():
                return False
        return True

    def put(self, item):
        """
        Adds elements to the priority queue
        :param item:
        :return:
        """
        prio, element = item[0], item[1]
        self.prio_dict[prio].put(element)

    def get(self):
        """
        Gets an Element from the PriorityQueue with the highest priority.
        :return:
        """
        if self.empty():
            return None

        prios = sorted(self.prio_dict.keys())
        for prio in prios:
            queue_at_prio = self.prio_dict[prio]
            if not queue_at_prio.empty():
                return prio, queue_at_prio.get()

    def to_list(self):
        list_queue = []
        for prio in self.prio_dict.keys():
            queue_at_prio = self.prio_dict[prio]
            elements = list(queue_at_prio.queue)
            list_queue.extend([(prio, element) for element in elements])
        return list_queue