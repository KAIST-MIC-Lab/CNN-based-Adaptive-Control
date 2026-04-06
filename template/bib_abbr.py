"""
bib_abbr.py
- INPUT_BIB: input .bib file path with full name of publishers
- OUTPUT_BIB: output .bib file path with abbreviations of publishers

OUTPUT:
- A new .bib file with abbreviations of publishers

VERSION HISTORY:
- Version 1.0 (Date: 2025.04.14)
    - Initial release of the bib_abbr.py.
    - Supports replacing full publisher names with their abbreviations in a .bib file.

- Myeongseok Ryu
- 2025.06.21
"""

import os
import re

INPUT_BIB = "template/refs.bib"  # Input .bib file path
CURRENT_DIR = os.getcwd()

publisher_abbr = {
    "Annual Reviews in Control": "Annu. Rev. Control",
    "Applied Sciences": "Appl. Sci.-Basel",
    "arXiv, Preprint": "arXiv",
    "Automatica": "Automatica",
    "Biological Cybernetics": "Biol. Cybern.",
    "Electronics": "Electronics",
    "Frontiers in Psychiatry": "Front. Psychiatry",
    "Harvard Data Science Review": "Harv. Data Sci. Rev.",
    "IEEE Access": "IEEE Access",
    "IEEE Control Systems Letters": "IEEE Control Syst. Lett.",
    "IEEE Robotics and Automation Letters": "IEEE Robot. Autom. Lett.",
    "IEEE Transactions on Automatic Control": "IEEE Trans. Autom. Control",
    "IEEE Transactions on Control Systems Technology": "IEEE Trans. Control Syst. Technol.",
    "IEEE Transactions on Fuzzy Systems": "IEEE Trans. Fuzzy Syst.",
    "IEEE Transactions on Industrial Electronics": "IEEE Trans. Ind. Electron.",
    "IEEE Transactions on Neural Networks": "IEEE Trans. Neural Netw.",
    "IEEE Transactions on Neural Networks and Learning Systems": "IEEE Trans. Neural Netw. Learn. Syst.",
    "IEEE Transactions on Systems, Man, and Cybernetics: Systems": "IEEE Trans. Syst. Man Cybern. Syst.",
    "IEEE Transactions on Transportation Electrification": "IEEE Trans. Transp. Electr.",
    "IEEE/ASME Transactions on Mechatronics": "IEEE/ASME Trans. Mechatron.",
    "IEE Proceedings F (Radar and Signal Processing)": "IEE Proc. Radar Signal Process.",
    "IFAC Proceedings Volumes": "IFAC Proc. Vol.",
    "International Journal of Information and Systems Sciences": "Int. J. Inf. Syst. Sci.",
    "JSME International Journal Series C": "JSME Int. J. Ser. C",
    "Machines": "Machines",
    "Mathematics of Control, Signals, and Systems (MCSS)": "Math. Control Signals Syst.",
    "NeuroImage": "Neuroimage",
    "Proceedings of the Institution of Mechanical Engineers, Part D": "Proc. Inst. Mech. Eng. D",
    "Proceedings of the Institution of Mechanical Engineers, Part K": "Proc. Inst. Mech. Eng. K",
    "SAE Transactions": "SAE Trans.",
    "{SAE} Technical Paper Series": "SAE Tech. Pap.",
    "TechRxiv, Preprint": "TechRxiv",
    "Vehicle System Dynamics": "Veh. Syst. Dyn.",
    "12th International Munich Chassis Symposium 2021": "Munich Chassis Symp. 2021",
    "13th International Symposium on Advanced Vehicle Control (AVEC’16)": "AVEC 2016",
    "14. Internationales Stuttgarter Symposium": "Int. Stuttgarter Symp.",
    "2004 43rd IEEE Conference on Decision and Control (CDC)": "Proc. IEEE Conf. Decis. Control (CDC)",
    "2019 IEEE 58th Conference on Decision and Control (CDC)": "Proc. IEEE Conf. Decis. Control (CDC)",
    "2021 60th IEEE Conference on Decision and Control (CDC)": "Proc. IEEE Conf. Decis. Control (CDC)",
    "2006 IEEE Conference on Computer Aided Control System Design, 2006 IEEE International Conference on Control Applications, 2006 IEEE International Symposium on Intelligent Control":
        "IEEE Conf. CACS, CCA, ISIC",
    "2013 Africon": "Africon",
    "2019 18th European Control Conference (ECC)": "Eur. Control Conf. (ECC)",
    "International Conference on Learning Representations": "Int. Conf. Learn. Represent. (ICLR)",
    "Proceedings of Thirty Third Conference on Learning Theory": "Conf. Learn. Theory (COLT), vol. 33",
    "Proceedings of the 34th International Conference on Neural Information Processing Systems": "Adv. Neural Inf. Process. Syst. 34 (NeurIPS)",
    "Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining": "Proc. 25th ACM SIGKDD Conf. Knowl. Discov. Data Min.",
    "Proceedings of The 33rd International Conference on Machine Learning": "Int. Conf. Mach. Learn. (ICML), vol. 33",
    "Proceedings of the International Conference on Machine Learning": "Int. Conf. Mach. Learn. (ICML)",
    "arXiv preprint arXiv:1705.05502": "arXiv:1705.05502",
    "Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining": "Proc. ACM SIGKDD Int. Conf. Knowl. Discov. Data Min."
}


def my_process(input_bib, output_bib):
    with open(input_bib, 'r', encoding='utf-8') as f:
        bib_text = f.read()

    # publishers in bib_test
    publishers = set()

    for line in bib_text.splitlines():
        if line.strip().startswith("journal") or line.strip().startswith("booktitle") or line.strip().startswith("howpublished"):

            # publisher name
            pub_name = re.search(r'=\s*{(.+)},\s*$', line)

            publishers.add(pub_name)

    # Report unique publishers
    len_pubs = len(publishers)
    print("\n")
    print(f"Found {len_pubs} unique publishers.")

    # Find non-defined abbreviations
    non_defined_abbr = set()
    for line in bib_text.splitlines():
        if line.strip().startswith("journal") or line.strip().startswith("booktitle") or line.strip().startswith("howpublished"):
            pub_name = re.search(r'=\s*{(.+)},\s*$', line)
            if pub_name:
                pub_name = pub_name.group(1)
                if pub_name not in publisher_abbr:
                    non_defined_abbr.add(pub_name)

    # Report non-defined abbreviations
    if non_defined_abbr:
        print("Non-defined abbreviations found:")
        for abbr in non_defined_abbr:
            print(f"  - {abbr}")
            print(f"You can add the abbreviation to the 'publisher_abbr' dictionary in the script.")
            print(f"You are highly recommended to use Chat GPT 🤷‍♂️")
            input("Proceed? (enter/ctrl-c): ")
    else:
        print("👍 No non-defined abbreviations found.")

    # Replace publisher names with abbreviations
    print("\n")
    print("Replacing full publisher names with abbreviations...")
    for full_name, abbr in publisher_abbr.items():
        print(f"  👨‍🔧 {full_name} 👉 {abbr}")
        bib_text = re.sub(re.escape(full_name), abbr, bib_text)
    
    # Write the modified content to a new file
    with open(output_bib, 'w', encoding='utf-8') as f:
        f.write(bib_text)

def main():
    print(f"""  
╔═══════════════════════════════════════════════╗
║          Bibliography Abbreviation Tool       ║
╠═══════════════════════════════════════════════╣
║ Developed by Myeongseok Ryu on June 21, 2025  ║
║ Contact: dding_98@kaist.ac.kr                 ║
║ Version 1.0 (Date: Jun 21, 2025)              ║   
╚═══════════════════════════════════════════════╝

DISCRIPTION:
  - This script processes a .bib file and replaces full publisher names
    with their abbreviations, as defined in the 'publisher_abbr' dictionary.
  - It searches publishers in the fileds 'journal', 'booktitle', and 'howpublished' of the .bib file.
          
Let's begin! (Your running in {CURRENT_DIR})
        """)
    
    input_bib = input(f"Enter the input .bib file path (default: {INPUT_BIB}): ")
    if input_bib == "":
        input_bib = INPUT_BIB
    output_bib = input_bib.replace(".bib", "_abbr.bib") 

    print(f"""
╔═══════════════════════════════════════════════╗
║                 Confirmation                  ║
╠═══════════════════════════════════════════════╣             
║ You have selected:                            ║
║  - Input .bib file: {input_bib}
╚═══════════════════════════════════════════════╝

Please confirm the above information is correct and press Enter to continue or Ctrl+C to exit.
""")    
    input("Press Enter to continue...")

    my_process(input_bib, output_bib)

    print("\n")
    print(f"Terminating this script. The output .bib file is saved as {output_bib}.")

if __name__ == "__main__":
    main()