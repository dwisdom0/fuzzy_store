This repository holds the code for the plots in my
[Fuzzy Store blog post](https://dwisdom.gitlab.io/post/fuzzy_store).

# Quickstart
1. Install everything

```shell
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements_torch.txt
```

2. Draw the plots
```shell
python draw_plots.py
```

I started doing this in torch so now it has a torch dependency.
It also uses Pandas.
It would be pretty easy to rewrite this in NumPy without those two.
I'm going to move on to other projects instead.
