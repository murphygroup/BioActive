import os
import glob
from pathlib import Path

def visualize_outputs(outdir, filetypes,
                 campaign_directory=Path("./campaignDirectory")):
    visualizations = glob.glob(os.getcwd()+os.path.sep+outdir+os.path.sep+filetypes)
    #print(visualizations)
    visualize(visualizations)

def find_specific_campaign_dir(campaign_directory, outdir_glob):
    try:
        outdir = next(iter(
            dir_ for dir_ in campaign_directory.glob(outdir_glob) if dir_.is_dir()
        ))
    except StopIteration:
        pass
    else:
        return outdir

def visualize(fps):
    from IPython.display import Image, display
    from IPython.core.display import HTML
    html_snippets = [
        f"<h3>{fp}<h3>" + "\n" + "<img src=\"" + fp + "\">"
        for fp in fps]
    #print(html_snippets)
    display(HTML("\n".join(html_snippets)))
    return
