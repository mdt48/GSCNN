"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

# File taken from https://github.com/mcordts/cityscapesScripts/
# License File Available at:
# https://github.com/mcordts/cityscapesScripts/blob/master/license.txt

# ----------------------
# The Cityscapes Dataset
# ----------------------
#
#
# License agreement
# -----------------
#
# This dataset is made freely available to academic and non-academic entities for non-commercial purposes such as academic research, teaching, scientific publications, or personal experimentation. Permission is granted to use the data given that you agree:
#
# 1. That the dataset comes "AS IS", without express or implied warranty. Although every effort has been made to ensure accuracy, we (Daimler AG, MPI Informatics, TU Darmstadt) do not accept any responsibility for errors or omissions.
# 2. That you include a reference to the Cityscapes Dataset in any work that makes use of the dataset. For research papers, cite our preferred publication as listed on our website; for other media cite our preferred publication as listed on our website or link to the Cityscapes website.
# 3. That you do not distribute this dataset or modified versions. It is permissible to distribute derivative works in as far as they are abstract representations of this dataset (such as models trained on it or additional annotations that do not directly include any of our data) and do not allow to recover the dataset or something similar in character.
# 4. That you may not use the dataset or any derivative work for commercial purposes as, for example, licensing or selling the data, or using the data with a purpose to procure a commercial gain.
# 5. That all rights not expressly granted to you are reserved by us (Daimler AG, MPI Informatics, TU Darmstadt).
#
#
# Contact
# -------
#
# Marius Cordts, Mohamed Omran
# www.cityscapes-dataset.net

"""

from collections import namedtuple


#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )

#--------------------------------------------------------------------------------
# SUN Labels
#--------------------------------------------------------------------------------
import os
import numpy as np

scenes = ["alcove", "atrium home", "attic", "barrack", "basemaent", "bathouse", "bathroom", 
            "bedroom", "berth deck", "bow window indoor", "cabin indoor", "childs room", "closet",
            "dinette home", "dorm room", "dining room", "furnace room", "game room", "garage indoor",
            "home office", "home theater", "hotel room", "kitchen", "kitchenette", "lavatory",
            "living room" , "loft", "nursery", "parlor", "playroom", "poolroom home", "recreation room",
            "restaurant kitchen", "root cellar", "shower", "shower room", "staircase", "television room",
            "hotel", "hotel_room", "house"]

path = "./SUN2012/Annotations"

img_xml = []

for root, dirs, files in os.walk(path):
    for d in dirs:   
        if d in scenes:
            new_path = os.path.join(root, d)
            img_xml = [os.path.join(new_path, f) for f in os.listdir(new_path)]

import xml.etree.ElementTree as ET
classes = []
for xml in img_xml:
    tree = ET.parse(xml)
    root = tree.getroot()

    for child in root:
        child_nodes = {x.text for x in root.findall(child.tag+"/*") if x.tag == "name"}
        #print(child_nodes)
        for n in child_nodes:
            if n.strip() not in classes:
                classes.append(n.strip())

import math
from PIL import Image
im = Image.new('RGB', (604, 62))
ld = im.load()

def gaussian(x, a, b, c, d=0):
    return a * math.exp(-(x - b)**2 / (2 * c**2)) + d

for x in range(im.size[0]):
    r = int(gaussian(x, 158.8242, 201, 87.0739) + gaussian(x, 158.8242, 402, 87.0739))
    g = int(gaussian(x, 129.9851, 157.7571, 108.0298) + gaussian(x, 200.6831, 399.4535, 143.6828))
    b = int(gaussian(x, 231.3135, 206.4774, 201.5447) + gaussian(x, 17.1017, 395.8819, 39.3148))
    for y in range(im.size[1]):
        ld[x, y] = (r, g, b)

labels = [Label(c, idx, 255,"indoor", 0, True, False, tuple((np.random.choice(range(256), size=3)))) for idx, c in enumerate(classes)]

#--------------------------------------------------------------------------------
# Create dictionaries for a fast lookup
#--------------------------------------------------------------------------------

# Please refer to the main method below for example usages!

# name to label object
name2label      = { label.name    : label for label in labels           }
# id to label object
id2label        = { label.id      : label for label in labels           }
# trainId to label object
trainId2label   = { label.trainId : label for label in reversed(labels) }
# label2trainid
label2trainid   = { label.id      : label.trainId for label in labels   }
# trainId to label object
trainId2name   = { label.trainId : label.name for label in labels   }
trainId2color  = { label.trainId : label.color for label in labels      }
# category to list of label objects
category2labels = {}
for label in labels:
    category = label.category
    if category in category2labels:
        category2labels[category].append(label)
    else:
        category2labels[category] = [label]

#--------------------------------------------------------------------------------
# Assure single instance name
#--------------------------------------------------------------------------------

# returns the label name that describes a single instance (if possible)
# e.g.     input     |   output
#        ----------------------
#          car       |   car
#          cargroup  |   car
#          foo       |   None
#          foogroup  |   None
#          skygroup  |   None
def assureSingleInstanceName( name ):
    # if the name is known, it is not a group
    if name in name2label:
        return name
    # test if the name actually denotes a group
    if not name.endswith("group"):
        return None
    # remove group
    name = name[:-len("group")]
    # test if the new name exists
    if not name in name2label:
        return None
    # test if the new name denotes a label that actually has instances
    if not name2label[name].hasInstances:
        return None
    # all good then
    return name

#--------------------------------------------------------------------------------
# Main for testing
#--------------------------------------------------------------------------------

# just a dummy main
if __name__ == "__main__":
    # Print all the labels
    print("List of sun labels:")
    print("")
    print(("    {:>21} | {:>3} | {:>7} | {:>14} | {:>10} | {:>12} | {:>12}".format( 'name', 'id', 'trainId', 'category', 'categoryId', 'hasInstances', 'ignoreInEval' )))
    print(("    " + ('-' * 98)))
    for label in labels:
        print(("    {:>21} | {:>3} | {:>7} | {:>14} | {:>10} | {:>12} | {:>12}".format( label.name, label.id, label.trainId, label.category, label.categoryId, label.hasInstances, label.ignoreInEval )))
    print("")

    # print("Example usages:")

    # Map from name to label
    # name = 'car'
    # id   = name2label[name].id
    # print(("ID of label '{name}': {id}".format( name=name, id=id )))

    # # Map from ID to label
    # category = id2label[id].category
    # print(("Category of label with ID '{id}': {category}".format( id=id, category=category )))

    # # Map from trainID to label
    # trainId = 0
    # name = trainId2label[trainId].name
    # print(("Name of label with trainID '{id}': {name}".format( id=trainId, name=name )))
