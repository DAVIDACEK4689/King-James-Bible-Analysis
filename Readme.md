<h1 style="text-align: center;">Data Science Project</h1>
<h2 style="text-align: center;">King James Bible</h2>

<h2>Description</h2>
<p style="text-align: justify;">
    This repository contains a simple data science project with analysis of the King James Bible dataset, especially focused on Torah.
</p>

<h2>Usage</h2>

<h3>Offline</h3>
<p style="text-align: justify;">
    The offline version of the report with expandable code blocks is available in <a href="report.md">report.md</a>.
</p>

<h3>Online</h3>
<p style="text-align: justify;">
    The Jupyter notebook is available in <a href="report.ipynb">report.ipynb</a>, with all the requirements listed in <a href="requirements.txt">requirements.txt</a>.
</p>

<details>
<summary>Click to expand</summary>


```python
import pandas as pd

# Define a function to parse the book, chapter, and verse
def parse_reference(reference):
    book, chapter_verse = reference.rsplit(' ', 1)
    chapter, verse = map(int, chapter_verse.split(':'))
    return book, chapter, verse

# Load the file
file_path = "data/bible.txt"
data = pd.read_csv(file_path, header=None, sep="\t", names=["Book Chapter:Verse", "Text"])
data["Book"], data["Chapter"], data["Verse"] = zip(*data["Book Chapter:Verse"].apply(parse_reference))
data = data[["Book", "Chapter", "Verse", "Text"]]

# Display the resulting DataFrame
data.head()
```
</details>