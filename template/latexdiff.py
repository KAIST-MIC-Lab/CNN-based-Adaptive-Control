"""
latexdiff.py
- COMMIT1: new commit hash.
- COMMIT2: old commit hash.

OUTPUT:
- COMMIT1_COMMIT2.pdf

VERSION HISTORY:
- Version 1.0 (Date: 2025.04.14)
    - Initial release of the LaTeX diff tool.
    - Supports comparing LaTeX documents across Git commits.
    - Generates a PDF file showing the differences.
- Version 1.1 (Date: 2025.05.25)
    - fix some bugs.
- Version 1.2 (Date: 2025.06.08)
    - The output file name is now COMMIT1_COMMIT2.pdf
    - So, you can recognize which commits are compared.
- Version 1.3 (Date: 2025.06.23)
    - When fail to finish the program, diff.tex will remain in the current directory.
    - You can compile it manually.
- Version 2.0 (Date: 2025.08.07)
    - Windows compatibility.
- Myeongseok Ryu
- dding_98@kaist.ac.kr
- 2025.04.14
"""

import os
import subprocess
import glob
import shutil

TEX_FILE_NAME = "manuscript.tex"
CURRENT_DIR = os.getcwd()
SAVE_DIR = "."

def run_terminal_command(command, shell=True):
    print(f"$ {command}")
    result = subprocess.run(command, shell=shell)
    if result.returncode != 0:
        raise RuntimeError(f"Command '{command}' failed with exit code {result.returncode}.")

def compile_tex(file_name):
    run_terminal_command(f"pdflatex -interaction=batchmode {file_name}")
    run_terminal_command(f"bibtex {file_name[:-4]}.aux")
    run_terminal_command(f"pdflatex -interaction=batchmode {file_name}")
    run_terminal_command(f"pdflatex -interaction=batchmode {file_name}")

def clean_up():
    print("Cleaning up...")

    patterns = [
        "tmp1.tex", "tmp2.tex", "tmp1.pdf", "tmp2.pdf",
        "*.aux", "*.log", "*.out", "*.bbl", "*.blg", "*.run.xml",
        "*.toc", "*.synctex.gz", "*.fdb_latexmk", "*.fls", "*.spl", "*.dvi"
    ]

    for pattern in patterns:
        for file_path in glob.glob(os.path.join(CURRENT_DIR, pattern)):
            try:
                os.remove(file_path)
            except Exception:
                pass

    print("Cleanup complete.")

def main():
    print(f"""  
╔═══════════════════════════════════════════════╗
║             LaTeXDiff Visualizer              ║
║         Git-based LaTeX Difference Tool       ║
╠═══════════════════════════════════════════════╣
║ Developed by Myeongseok Ryu on April 14, 2025 ║
║ Contact: dding_98@kaist.ac.kr                 ║
║ Version 2.0 (Date: August 07, 2025)           ║
╚═══════════════════════════════════════════════╝

DISCRIPTION:
  - Compare your LaTeX documents across Git commits
    and visualize the changes with elegance.
          
Let's begin! (You're running in {CURRENT_DIR})
""")

    try:
        tex_file_name = input(f"Enter the tex file name (default: {TEX_FILE_NAME}): ").strip() or TEX_FILE_NAME

        print("""
OPTIONS:
  - r: current working tree (unsaved changes)
  - h: HEAD (latest commit)
  - p: previous commit of selected one
""")

        commit1 = input("Enter the first commit hash of new one (r/h/SHA): ").strip()
        if not commit1:
            raise ValueError("Please enter the commit hash.")
        elif commit1 == "p":
            raise ValueError("option p is not available for the first commit.")
        elif commit1 == "h":
            commit1 = "HEAD"

        commit2 = input("Enter the second commit hash of old one (h/p/SHA): ").strip()
        if not commit2:
            raise ValueError("Please enter the second commit hash.")
        elif commit2 == "p":
            base_commit = "HEAD" if commit1 == "r" else commit1
            commit2 = subprocess.check_output(["git", "rev-parse", f"{base_commit}^"]).decode().strip()
        elif commit2 == "h":
            commit2 = "HEAD"

        print(f"""
╔═══════════════════════════════════════════════╗
║                 Confirmation                  ║
╠═══════════════════════════════════════════════╣             
║ You have selected:                            ║
║  - New commit: {commit1}                       
║  - Old commit: {commit2}                       
║  - LaTeX file: {tex_file_name}                
╚═══════════════════════════════════════════════╝

Please confirm the above information is correct and press Enter to continue or Ctrl+C to exit.
""")
        input("Press Enter to continue...")

        # checkout the commit
        if commit1 == "r":
            shutil.copyfile(tex_file_name, "tmp1.tex")
        else:
            content1 = subprocess.check_output(["git", "show", f"{commit1}:{tex_file_name}"]).decode()
            with open("tmp1.tex", "w", encoding='utf-8') as f:
                f.write(content1)

        content2 = subprocess.check_output(["git", "show", f"{commit2}:{tex_file_name}"]).decode()
        with open("tmp2.tex", "w", encoding='utf-8') as f:
            f.write(content2)

        # create diff.tex
        run_terminal_command("latexdiff --flatten tmp2.tex tmp1.tex > diff.tex")
        compile_tex("diff.tex")

        save_PDF = f"diff_{commit1[:6]}_{commit2[:6]}.pdf"
        shutil.move("diff.pdf", save_PDF)

        print("\nSuccessfully generated the diff.tex file and compiled it to PDF.")
        clean_up()

        print(f"""
╔═══════════════════════════════════════════════╗
║             Successfully Generated!           ║
╠═══════════════════════════════════════════════╣             
║ The LaTeX diff PDF file is ready:             ║
║  - {CURRENT_DIR}\\{save_PDF}
║ All temporary files have been cleaned up.     ║
╚═══════════════════════════════════════════════╝
""")

        print("Done!")

    except Exception as e:
        print(f"""
╔═══════════════════════════════════════════════╗
║         Failed to Generate LaTeX Diff!        ║
╠═══════════════════════════════════════════════╣             
║ An error occurred:                            ║   
║   - {e}                                      
║ All temporary files have been cleaned up.     ║
╠═══════════════════════════════════════════════╣
║ Please check the input and try again.         ║
║ If the problem persists, please contact the   ║
║ developer at the email below.                 ║
║   - Myeongseok Ryu (dding_98@kaist.ac.kr)     ║
╚═══════════════════════════════════════════════╝
""")

        save_tex = f"diff_{commit1[:6]}_{commit2[:6]}.tex"
        shutil.move("diff.tex", save_tex)
        print(f"Saved the diff file as {save_tex} for manual compilation.")

        clean_up()

if __name__ == "__main__":
    main()