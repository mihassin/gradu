import os


# Return path to /gradu folder
def get_project_root():
	path_str = os.path.abspath(__file__)
	path_list = str.split(path_str, '/')
	path = '/'.join(path_list[:-2])
	return path


# Lists the immidiate subdirectories below the given path
def immidiate_subdirs(root):
	return [f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))]	
