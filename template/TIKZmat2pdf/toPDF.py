import os

SRC_DIFF = "../src/script_simulation/figures/compare"
TARGET_DIR = "../src/script_simulation/figures/compare"

fig_num = 11

try:
    for i in range(1, fig_num + 1):
        print(f"Converting Fig{i} to PDF...")

        tmp_file_name = f"tmp_fig_{i}"

        with open(f"{tmp_file_name}.tex", "w") as tex_file:

            tex_content = f"""
    \\documentclass{{standalone}}

    \\usepackage{{pgfplots}}
    \\pgfplotsset{{compat=newest}}
    %% the following commands are needed for some matlab2tikz features
    % \\usetikzlibrary{{plotmarks}}
    % \\usetikzlibrary{{arrows.meta}}
    % \\usepgfplotslibrary{{patchplots}}
    % \\usepackage{{grffile}}
    % \\usepackage{{amsmath}}

    \\usepgfplotslibrary{{external}} 
    \\tikzexternalize

    %% you may also want the following commands
    %   \\pgfplotsset{{plot coordinates/math parser=false}}
    %   \\newlength\\figureheight
    %   \\newlength\\figurewidth

    \\begin{{document}}
    \\input{{{SRC_DIFF}/Fig{i}.tex}}
    \\end{{document}}
            """
            tex_file.write(tex_content)
            tex_file.close()

        print(f"Compiling {tmp_file_name}.tex to PDF...")
        os.system(f"lualatex -shell-escape -interaction=nonstopmode -halt-on-error {tmp_file_name}.tex > /dev/null 2>&1")
        os.system(f"mv {tmp_file_name}-figure0.pdf {TARGET_DIR}/Fig{i}.pdf")
        print(f"PDF file {TARGET_DIR}/Fig{i}.pdf created successfully.")

        print(f"Cleaning up temporary files...")
        os.system(f"rm -f {tmp_file_name}.tex")
        os.system(f"rm -f {tmp_file_name}.aux")
        os.system(f"rm -f {tmp_file_name}.log")
        os.system(f"rm -f {tmp_file_name}.auxlock")
        os.system(f"rm -f {tmp_file_name}-figure0.dpth")
        os.system(f"rm -f {tmp_file_name}-figure0.log")
        os.system(f"rm -f {tmp_file_name}-figure0.md5")
        os.system(f"rm -f {tmp_file_name}.pdf")
        print(f"Temporary files cleaned up.")
        print(f"Done with Fig{i}.\n")

    print("All figures converted to PDF successfully.")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    print(f"Cleaning up temporary files...")
    os.system(f"rm -f {tmp_file_name}.tex")
    os.system(f"rm -f {tmp_file_name}.aux")
    os.system(f"rm -f {tmp_file_name}.log")
    os.system(f"rm -f {tmp_file_name}.auxlock")
    os.system(f"rm -f {tmp_file_name}-figure0.dpth")
    os.system(f"rm -f {tmp_file_name}-figure0.log")
    os.system(f"rm -f {tmp_file_name}-figure0.md5")
    os.system(f"rm -f {tmp_file_name}.pdf")
    print(f"Temporary files cleaned up.")
    print("Script finished.")