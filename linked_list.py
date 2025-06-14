# linked list using OOP

class Node:
    def __init__(self, data):
        self.data = data
        self.next = None 


class LinkedList:
    def __init__(self):
        self.head = None

    def add_node(self, data):
        # Adds a node to the end
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
            print(f"Added {data} as the first node.")
        else:
            temp = self.head
            while temp.next is not None:
                temp = temp.next
            temp.next = new_node
            print(f"Added {data} at the end.")

    def print_list(self):
        # Prints the linked list
        if self.head is None:
            print("List is empty.")
            return

        temp = self.head
        while temp:
            print(temp.data, end=" -> ")
            temp = temp.next
        print("None")

    def delete_nth_node(self, n):
        # Deletes the nth node
        if self.head is None:
            print("Error: Cannot delete from an empty list.")
            return

        if n <= 0:
            print("Error: Invalid position. Use a positive index.")
            return

        if n == 1:
            print(f"Deleted node at position {n}.")
            self.head = self.head.next
            return

        temp = self.head
        prev = None
        count = 1

        while temp is not None and count < n:
            prev = temp
            temp = temp.next
            count += 1

        if temp is None:
            print("Error: Index out of range.")
            return

        prev.next = temp.next
        print(f"Deleted node at position {n}.")

# Testing the LinkedList

if __name__ == "__main__":
    ll = LinkedList()

    # Adding nodes
    ll.add_node(5)
    ll.add_node(10)
    ll.add_node(15)
    ll.add_node(20)

    # Displaying list
    ll.print_list()

    # Deleting 3rd node
    ll.delete_nth_node(3)
    ll.print_list()

    # Deleting head node
    ll.delete_nth_node(1)
    ll.print_list()

    # delete node out of range
    ll.delete_nth_node(10)

    # delete from empty list
    ll.delete_nth_node(1)
    ll.delete_nth_node(1)
    ll.delete_nth_node(1)
