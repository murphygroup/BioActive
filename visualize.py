#import os
#import glob
#from pathlib import Path

def visualize(fps):
    from IPython.core.display import display, HTML
    html_snippets = [
        f"<h3>{fp.as_posix()}</h3>" + "<img src=\"" + fp.as_posix() + "\">"
        for fp in fps]
    print(html_snippets)
    display(HTML("\n".join(html_snippets)))
    return
