import os
from os.path import join
import subprocess
import shutil
from pygments import highlight
from pygments.lexers import DiffLexer
from pygments.formatters import HtmlFormatter

from vla_network.utils.logger import log
from vla_network.utils.utils import VLA_NETWORK_ROOT


def get_git_status(command, path=VLA_NETWORK_ROOT):
    try:
        git_command = dict(
            status=["git", "status"],
            diff=["git", "diff"],
            diff_staged=["git", "diff", "--staged"],
            id=["git", "rev-parse", "HEAD"],
            untracked=["git", "ls-files", "--others", "--exclude-standard"],
        )[command]
        result = subprocess.check_output(git_command, cwd=path)
        return result.decode("utf-8")
    except subprocess.CalledProcessError as e:
        print(f"Error getting current commit ID: {e}")
        return None


def generate_untracked_diff(untracked_files, path=VLA_NETWORK_ROOT):
    diff_lines = []
    for file in untracked_files:
        file_path = os.path.join(path, file)
        if os.path.isfile(file_path):
            try:
                with open(file_path, "r") as f:
                    content = f.read()
            except Exception as e:
                content = f"Error reading untracked file {file}: {e}"
            diff_lines.append(f"diff --git a/{file} b/{file}")
            diff_lines.append("new file mode 100644")
            diff_lines.append("index 0000000..0000000")  # Dummy hash
            diff_lines.append("--- /dev/null")
            diff_lines.append(f"+++ b/{file}")
            line_count = len(content.splitlines())
            diff_lines.append(f"@@ -0,0 +1,{line_count if line_count > 0 else 1} @@")
            for line in content.splitlines():
                diff_lines.append(f"+{line}")
    return "\n".join(diff_lines)


def save_diff_with_syntax_highlighting(diff_text, output_file):
    lexer = DiffLexer()
    formatter = HtmlFormatter(full=True, linenos=True)
    highlighted_diff = highlight(diff_text, lexer, formatter)

    with open(output_file, "w") as f:
        f.write(highlighted_diff)


def save_all_python_files(directory, output_file):
    # remove the output_file directory
    if os.path.exists(output_file):
        shutil.rmtree(output_file)
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py") or file.endswith(".json"):
                file_path = os.path.join(root, file)
                # copy file
                target_path = join(output_file, file_path[len(directory) + 1 :])
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                shutil.copy(file_path, target_path)
    log.info("All python files have been saved")


def save_code_and_git(exp_root: str):
    with open(join(exp_root, "git_status.txt"), "w") as f:
        f.write(f"Commit ID: {get_git_status('id')}\n")
        f.write("\n")
        f.write("\n")
        f.write(get_git_status("status"))
        f.write("\n")
        f.write("\n")
        f.write(get_git_status("diff"))
        f.write("\n")
        f.write("\n")
        f.write(get_git_status("diff_staged"))
    # Generate diff with untracked files
    diff_text = get_git_status("diff")
    untracked_files = get_git_status("untracked").splitlines()
    if untracked_files:
        untracked_diff = generate_untracked_diff(untracked_files)
        diff_text += "\n" + untracked_diff

    save_diff_with_syntax_highlighting(diff_text, join(exp_root, "diff.html"))
    save_diff_with_syntax_highlighting(
        get_git_status("diff_staged"), join(exp_root, "diff_staged.html")
    )
    save_all_python_files(VLA_NETWORK_ROOT, join(exp_root, "code"))
