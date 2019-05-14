'''
Utility to copy files from one folder to another
'''
import os
import shutil

cwd = "C:/Users/ashok/OneDrive/Desktop/[FCO] AppliedAICourse - Applied Machine Learning Course"
dest = "C:/Users/ashok/OneDrive/Desktop/Applied AI"
entries = os.listdir(cwd)
print(entries)

#for moving files from one directory to another
file_path_list = []
for entry in entries:
    current_path = os.path.join(cwd,entry,'')
    files = os.listdir(current_path)
    # print(files)
    for file in files:
        if file.endswith('.mkv'):
            print(file, " is about to be copied")
            shutil.move(current_path+"/"+file,dest)
            print(("File copy is done"))
#             # file_path_list.append(current_path+"/"+file)
#             # print(file)


# file_path_list = []
# for entry in entries:
#     current_path = os.path.join(cwd,entry,'')
#     files = os.listdir(current_path)
#     # print(files)
#     for file in files:
#         if file.endswith('.pdf'):
#             print(file, " is about to be copied")
#             # print(current_path[88:].strip('\\'))
#             os.rename(current_path+'/'+'out.pdf',str(current_path[88:].strip('\\'))+'.pdf')
#             # shutil.move(current_path+"/"+file,dest)
#             # print(("File copy is done"))
#             # file_path_list.append(current_path+"/"+file)
#             # print(file)

# print("Printing complete List",file_path_list)
# for i in range(10):
#     print(file_path_list[i])




