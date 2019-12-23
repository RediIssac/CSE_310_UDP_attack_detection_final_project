"""create a new csv file with the rows from test.csv with the correct number of columns"""

import csv

def ip_to_seq(csv_list):
    for i in range(len(csv_list)):
        if(i == 1 or i == 2):
          ip_list = csv_list[i].split(".")
          for j in range(len(ip_list)):
            numberOfZeros= 3-len(ip_list[j])
            ip_list[j] = str(numberOfZeros * '0')+ip_list[j]
            ip_seq = "".join(ip_list)
            csv_list[i] = ip_seq
        if(i == 4):
          if csv_list[4] == "64\n":
              csv_list[4] = "0\n"
          else:
              csv_list[4] = "1\n"

# def change_labels(csv_list):
#     if csv_list[4] == 64:
#         csv_list[4] = "0"
#     else:
#         csv_list[4] = "1"

def create_clean_csv(test_arr, length):
    with open('clean_test.csv', 'w') as csvfile:
        for row in test_arr:
            csv_list = row.split(",")
            if len(csv_list) == length:
                ip_to_seq(csv_list)
                # change_labels(csv_list) 
                csvfile.write(",".join(csv_list))

if __name__ == "__main__":
    with open("test.csv") as test_file:
        test_arr = test_file.readlines()
        create_clean_csv(test_arr, 5)