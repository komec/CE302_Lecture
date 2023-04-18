# Week 2
# Quick Intro. to Python

# Data Types

# Numbers

x = 2j + 4
type(x)

# Strings

y = "hi AI"
type(y)

# Boolean
type(5 == 4)

# Lists
z = [2, 6, 8, 1, "hello", ['Gamze'], True, 3.171]
type(z)

# Tuple
d = (2, 5, 7, " ", 3.18)
type(d)                      # WHY ??????

# Dictionary
d = {'Christian': ["America",18],
        'Daisy':["England",12],
        'Antonio':["Spain",22],
        'Dante':["Italy",25]}

type(d)

# set
s = {'Christian', "America",18,'Daisy',"England",12}
type(s)


# FUNCTIONS ________________________________________________________

print("a", "b", sep="___")


def calculate( x ):
    print(x * 2)

calculate( 5)

# CONDITIONS ___________________________________________________________

# if /  else & elif / for loop


##  Exercises
#
#   1 )  Write a code that gives the type of variables below

x = 8
y = 3.2
z = 8j + 18
l = [1, 2, 3, 4]
t = ("Machine Learning", "Data Science")
s = {"Python", "Machine Learning", "Data Science"}
d = {"Name": "Jake",
     "Age": 27,
     "Adress": "Downtown"}
c = 23 < 22

# list1 = [x,y,z,l,t,s,d,c]
# for i in list1:
#     print(type(i))

[print("type is", type( i )) for i in list1]



#   2 )    Convert all letters of the given string to uppercase. Put space instead of commas and periods, separate them word by word.

txt = "Success is a journey, not a destination."
# [word.upper() for word in txt.split(" ")]
#print([*txt.upper()])
txt.upper().replace(",", " ").replace(".", " ").split()


#  3)   Complete the following tasks for the given list.

lst = ["A","D","V","A","N","C","E","D","P","R","O","G","R","A","M","I","N","G"]
# Step 1: Look at the number of elements of the given list.
print(len(lst))

# Step 2: Call the elements at index zero and ten.
print(lst[:10])

# Step 3: Create a list ["A","D","V","A","N","C","E","D"] from the given list.
print(lst[:8])

# Step 4: Delete the element in the eighth index.
del lst[7]
lst.pop(8)
print(lst)

# Step 5: Add a new element.
lst.append('X')
print(lst)

# Step 6: Re-add element "D" to the eighth index.
lst.insert(8, "D")
print(lst)


# 4 ) Complete the following tasks for the given dictionary.

clss = {"Eren" : ["Bayburt", 22],
        "Gamze":["Agri", 23],
        "Berkay":["Kocaeli", 21]}

# Step 1: Access the key values.
clss.keys()


# Step 2: Access the values.
clss.values()


# Step 3: Update the value 21 of the Berkay key to 45.
clss["Berkay"] = 45
print (clss)

# Step 4: Add a new value whose key value is Gamze value [Van,24].
clss['Gamze']=['Van',24]

# Step 5: Delete Eren from dictionary.
clss.pop("Eren")



# 5 )  Write a function that takes a list as an argument, assigns the odd and even numbers in the list to separate lists, and returns these lists.

lst = [13, 20, 3, 8 , 6 ,14, 5 , 23 , 19, 10, 32]

def separate_odd_even(lst):
    even_lst = []
    odd_lst = []
    for num in lst:
        if num % 2 == 0:
            even_lst.append(num)
        else:
            odd_lst.append(num)
    return even_lst, odd_lst

print(separate_odd_even(lst))



# 6 ) Three lists are given below. In the lists, there is a course code, credit and quota information, respectively.
#       Print course information using zip.

code = ["CE103", "CE302","EQE582"]
akts = [2, 4, 3]
stdn_num = [82, 15, 28]

print(list(zip(code,akts,stdn_num)))

# Alternatif :
# for code, akts, stdn_num in zip(code, akts, stdn_num):


# 7 ) In the list given below are the names of the students who received degrees in engineering and social sciences faculties.
#  Respectively, the first three students represent the success rank of the engineering faculty, while the last three students belong to the rank of the medical faculty.
#  Print student grades specific to faculty using enumarate.

stdn_name = ["Ayse" , "Neslihan","Aylin","Seyhun", "Mehmet", "Deniz"]

for i, x in enumerate(stdn_name):
    if i < 3:
        i += 1
        print("Eng", i, "Student Name", x)
    else:
        i -= 2
        print("Soc.Sci", i, "Student Name", x)

# for engineer_faculty in enumerate(stdn_name[0:3], 1):    print(engineer_faculty)
# for social_sciences in enumerate(stdn_name[3:6], 1):    print(social_sciences)


# 8 )  There are 2 sets given below.
#  If the 1st set includes the 2nd set, you are asked to define the function that will print the common elements of the 2nd set from the 1st set if it does not.

set_1 = set(["Ayse" , "Neslihan"])
set_2 = set(["Aylin","Seyhun", "Ayse" ,"Mehmet", "Deniz",  "Neslihan"])

def sets(set_1, set_2):
    if set_1.issuperset(set_2):
        print(set_1.intersection(set_2))
    else:
        print(set_2.difference(set_1))

sets(set_1,set_2)

# ALternatif 1
# same_word = []
# for word in set_1:
#     if word  in  set_2:
#         same_word.append(word)
# print(same_word)
