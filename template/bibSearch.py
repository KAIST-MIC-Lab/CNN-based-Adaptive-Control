"""
bibSearch.py
- search arguments: keywords, sort option, csv path, additional options

OUTPUT:
- searchResult.csv

- Myeongseok Ryu
- dding_98@kaist.ac.kr
- 2024.01.15
"""

import os
import subprocess
import sys
import shutil	

# INITIALIZE
CURRENT_DIR = os.getcwd()

def package_exists(pck):
	if shutil.which(pck) is None:
		print("Warning!")
		print(f"'{pck}' is not installed.")
		install = input("Do you want to install it? (y/n)")

		if install == 'y' or install == '':
			try:
				install_command = f"pip install {pck}"
				print(f"$ {install_command}")
				subprocess.check_call(install_command, shell=True)

			except subprocess.CalledProcessError as e:
				print(f"Error during installation: {e}")
				print("You can manually install it using the command:")
				print(f"$ pip install {pck}")
				print("\nTerminating the program.")
				sys.exit(1)

		elif install  == 'n':
			print("Installation aborted.")
			print("\nTerminating the program.")
			sys.exit(1)
		else:
			print("Invalid input. Please enter 'y' or 'n'.")
			print("\nTerminating the program.")
			sys.exit(1)

def main():
	PRE_KEYWORDS = "None"

	print(f"""
╔═══════════════════════════════════════════════╗
║           Bibliography Search Tool            ║
║       Google Scholar Search and Sorter        ║
╠═══════════════════════════════════════════════╣
║ Developed by Myeongseok Ryu on Jan. 15, 2024  ║
║ Contact: dding_98@kaist.ac.kr                 ║
║ Version 1.0 (Date: Jan 15, 2024)              ║   
╚═══════════════════════════════════════════════╝
	   
DISCRIPTION:
  - This script allows you to search Google Scholar
  - You can sort the search results by citation number, year, or citation/year ratio
  - You can save the results as a CSV file
          
Let's begin! (Your running in {CURRENT_DIR})
	""")

	package_exists("sortgs")

	# MAIN LOOP
	while True:
		
		print(f"""  
SEARCH OPTIONS:
  - hint: OR, AND, \"\", intitle, intext, author
  - previous keyword: {PRE_KEYWORDS} (default)
  - insert -9 to quit program
    """)

		keywords = input("Insert Keywords: ")

		if keywords == '':
			keywords = PRE_KEYWORDS	
		elif keywords == "-9":
			print("\nTerminating the program.")
			quit()
		else:
			PRE_KEYWORDS = keywords	
		
		print(f"""  
SORT OPTIONS:
  - 1:  citation number (default)
  - 2:  year
  - 3:  citation numver / year (ratio)
  - insert -9 to quit program
    """)

		sort_opt = input("Insert Sort Option: ")
		
		if sort_opt == '':
			sort_opt = "--sortbycit"
		elif int(sort_opt) == 1:
			sort_opt = "--sortbycit"
		elif int(sort_opt) == 2:
			sort_opt = "--sortby year"
		elif int(sort_opt) == 3:
			sort_opt = "--sortby cit/year"
		elif int(sort_opt) == -9:
			print("\nTerminating the program.")
			quit()
		else:
			print("err: wrong number")
			continue

		print(f"""
SAVE DIRECTORY:
  - Enter: default directory ~/desktop
  - -1: do not save
  - -9: quit program
	""")

		csvpath = input("Where to Save: ")

		if csvpath == '':
			csvpath = "--csvpath '~/desktop'"	
		elif csvpath == "-1":
			csvpath = "--notsavecsv"
		elif csvpath == "-9":
			print("\nTerminating the program.")
			quit()
		else:
			print("err: wrong path")
			continue

		print(f"""
ADDITIONAL OPTIONS:
  - hint: --startyear, --endyear
  - -1: do not add additional options
  - -9: quit program
	""")


		add_opt = input("If You Want: ")

		if add_opt == -9:
			print("\nTerminating the program.")
			quit()

		cmd = "sortgs "
		cmd += "'" + keywords + "' "
		cmd += sort_opt + " "
		cmd += csvpath + " "
		cmd += add_opt

		print("                                       ")
		print("TOTAL COMMAND: " + cmd)
		print("                                       ")
		tmp = os.system(cmd)
		
		print("Saved Search Result")
		tmp = input("Search Again?(y/n): ")
		
		if tmp == '':
			pass
		elif tmp == "n":
			print("\nTerminating the program.")
			quit()

	
if __name__ == "__main__":
    main()