import os
import queue


directories = ["../dataset/classify/white_check_pending/label/", "../dataset/classify/box/label/",
               "../dataset/classify/abstract/label/", "../dataset/classify/bottles/label/"]

object_directory_listing = []
object_set_listing = []
for dir in directories:
    object_dict = {}
    object_set = set()
    dir_files = os.listdir(dir)
    for f in dir_files:
        name = f.split("_")
        if name[0] not in object_dict:
            object_set.add(name[0])
            object_dict[name[0]] = 1
        else:
            object_dict[name[0]] += 1
    object_set_listing.append(object_set)
    object_directory_listing.append(object_dict)
print(object_directory_listing)
print("Objects that were totally filtered out: ", object_set_listing[0].difference(object_set_listing[1], object_set_listing[2], object_set_listing[3]))
filtered = [0, object_set_listing[0].intersection(object_set_listing[1]), object_set_listing[0].intersection(object_set_listing[2]), object_set_listing[0].intersection(object_set_listing[3])]

queue = queue.PriorityQueue()
for set in range(1, 4):
    for obj in filtered[set]:
        used = object_directory_listing[set][obj]
        removed = object_directory_listing[0][obj]
        fraction = removed/(used+removed)
        queue.put((fraction, removed, used, obj))
while not queue.empty():
    fraction, removed, used, obj = queue.get()
    print("Object " + obj + " used " + str(used) + ", removed " + str(removed) + ", fraction " + str(fraction))
