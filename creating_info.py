__author__ = 'StreetHustling'

file = open("car.info", "w")

file1 = open("bg.txt", "w")

for i in range(0, 550):
    file.write("pos/pos-"+str(i)+'.pgm 1 0 0 100 40 \n')


for i in range(0, 500):
    file1.write("neg1/neg-"+str(i)+'.pgm \n')