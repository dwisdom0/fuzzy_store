import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn.functional as F

from scipy.optimize import curve_fit
from tqdm import tqdm

from fuzzy_store import FuzzyStore

def exp_decay(t, base, lam, xoffset, yoffset):
  return base ** -((t + xoffset) * lam) + yoffset

def simple_hyperbola(x, scale, xoffset, yoffset):
  return yoffset + 1 / (scale * (x + xoffset))

def one_over_x_squared(x, scale, xoffset, yoffset):
  return yoffset + 1 / (scale * (x + xoffset)**2)

def inverse_sqrt(x, scale, xoffset, yoffset):
  return yoffset + 1 / (scale * np.sqrt(x + xoffset))

def plot_accuracy(df: pd.DataFrame, title: str=""):
  fig = go.Figure()

  # add the error bars first
  # based on
  # https://plotly.com/python/continuous-error-bars/
  df['lower_bound'] = df['cosine_sim'] - (df['std'] * 2)
  df['upper_bound'] = df['cosine_sim'] + (df['std'] * 2)

  colors = [
    'rgba(0, 0, 0, 0.15)',
    'rgba(0, 0, 200, 0.15)',
    'rgba(200, 0, 0, 0.15)',
  ]

  for i, d in enumerate(sorted(df['d'].unique())):
    error_df = df.loc[df['d']==d]
    fig.add_trace(go.Scatter(
      x=list(error_df['n']) + list(error_df['n'])[::-1],
      y=list(error_df['upper_bound']) + list(error_df['lower_bound'])[::-1],
      fill='toself',
      fillcolor=colors[i],
      line={'color': 'rgba(255,255,255,0)'},
      legendgroup=int(d),
      showlegend=False,
      name=f'95% CI (+/-2σ)'
    ))

  # add the actual lines on top so they're hoverable
  for i, d in enumerate(sorted(df['d'].unique())):
    line_df = df.loc[df['d']==d]
    fig.add_trace(go.Scatter(
      x=line_df['n'],
      y=line_df['cosine_sim'],
      line={'color': saturate(colors[i])},
      legendgroup=int(d),
      name=str(d)
    ))


  fig.update_layout(
    title=title,
    legend_title="Size of Fuzzy Store<br>(<i>d</i> in the <i>d</i>x<i>d</i> matrix)",
    xaxis_title="Number of key-value pairs stored",
    yaxis_title="Retrieval accuracy (cosine similarity)"
  )

  fig.show()


def saturate(rgba):
  values = rgba.replace('rgba(', '').replace(')', '')
  values = [int(val) for val in values.split(',')[:3]]
  return f'rgba({values[0]}, {values[1]}, {values[2]}, 1)'


def plot_curve_fits(df: pd.DataFrame):
  # initial plot
  ds = sorted(list(df['d'].unique()))
  colors = [
    'rgba(0, 0, 0, 1)',
    'rgba(0, 0, 200, 1)',
    'rgba(200, 0, 0, 1)',
  ]

  fig = go.Figure()

  for i, d in enumerate(ds):
    xs = df.loc[df['d']==d]['n'].to_numpy()
    ys = df.loc[df['d']==d]['cosine_sim'].to_numpy()
    # add an extra point at (1,1)
    xs = [1] + list(xs)
    ys = [1] + list(ys)

    fig.add_trace(go.Scatter(
      x=xs,
      y=ys,
      name=f"{d}",
      line=dict(
        color=colors[i],
        dash="dot",
      )
    ))

  # Exponential
  base_guess = 1.0
  lambda_guess = 0.01
  xoffset_guess = 0.0
  yoffset_guess = 0.0
  exp_params = [
    base_guess,
    lambda_guess,
    xoffset_guess,
    yoffset_guess
  ]

  fig = add_fit(
    df,
    fig,
    fit_func = exp_decay,
    init_params=exp_params,
    name="Exponential<br>(1/e<sup>x</sup>)",
    legendgroup=0,
    grayed_out=True,
  )

  # Inverse Square
  scale_guess = 0.002
  xoffset_guess = 0.0
  yoffset_guess = 0.25

  squared_params = [
    scale_guess,
    xoffset_guess,
    yoffset_guess,
  ]

  fig = add_fit(
    df,
    fig,
    fit_func=one_over_x_squared,
    init_params=squared_params,
    name="Inverse<br>Square<br>(1/x<sup>2</sup>)",
    legendgroup=2,
    grayed_out=True,
  )

  # Hyperbolic
  scale_guess = 0.002
  xoffset_guess = 0.0
  yoffset_guess = 0.25

  hyp_params = [
    scale_guess,
    xoffset_guess,
    yoffset_guess
  ]

  fig = add_fit(
    df,
    fig,
    fit_func=simple_hyperbola,
    init_params=hyp_params,
    name="Hyperbolic<br>(1/x)",
    legendgroup=1,
    grayed_out=True
  )

  # Inverse Square Root
  scale_guess = 0.002
  xoffset_guess = 0.0
  yoffset_guess = 0.25

  sqrt_params = [
    scale_guess,
    xoffset_guess,
    yoffset_guess
  ]

  fig = add_fit(
    df,
    fig,
    fit_func=inverse_sqrt,
    init_params=hyp_params,
    name="Inverse<br>Square Root<br>(1/√x)",
    legendgroup=3,
    grayed_out=True
  )


  fig.update_layout(
    title="Can you find the right fit function?<br><sup>Click on legend entries to overlay different fits</sup>",
    xaxis_title="Number of key-value pairs stored",
    yaxis_title="Retrieval accuracy (cosine similarity)",
  )

  fig.show()


def add_fit(df, fig, fit_func, init_params, name, legendgroup, grayed_out=False, visible="legendonly"):

  ds = sorted(list(df['d'].unique()))

  colors = [
    'rgba(0, 0, 0, 0.75)',
    'rgba(0, 0, 200, 0.75)',
    'rgba(200, 0, 0, 0.75)',
  ]

  for i, d in enumerate(ds):
    xs = df.loc[df['d']==d]['n'].to_numpy()
    ys = df.loc[df['d']==d]['cosine_sim'].to_numpy()
    # add an extra point at (1,1)
    xs = [1] + list(xs)
    ys = [1] + list(ys)

    fit = curve_fit(
      fit_func,
      xs,
      ys,
      p0=init_params
    )

    fit_params = fit[0]
    fit_xs = np.logspace(0, np.log10(np.max(xs)), 100)

    if grayed_out:
      color = "rgba(100, 100, 100, 0.75)"
    else:
      color = colors[i]

    if i == 0:
      showlegend = True
    else:
      showlegend = False


    fig.add_trace(go.Scatter(
      x=fit_xs,
      y=[fit_func(x, *fit_params) for x in fit_xs],
      name=name,
      line={"color": color},
      legendgroup=legendgroup,
      showlegend=showlegend,
      visible=visible
    ))

  return fig


def build_dfs():
  ds = [10, 100, 1000]
  ns = [int(x) for x in torch.logspace(1, 4, steps=30)]
  # add in the ds so that I can measure accuracy exactly at those points
  ns = sorted(ns + ds)

  records_random = []
  records_ortho = []
  for d in tqdm(ds, ascii=True, desc=f"Testing Fuzzy Stores"):
    for n in tqdm(ns, ascii=True, desc=f"{d=}, n: {min(ns)} - {max(ns)}"):
      new_record_random = [d, n]
      new_record_ortho = [d, n]
      fs_random = FuzzyStore(d, use_orthogonal_keys=False)
      fs_ortho = FuzzyStore(d, use_orthogonal_keys=True)
      python_dict_random = {}
      python_dict_ortho = {}
      # get n normalized vectors of size d
      # p=2 means L2 norm
      # https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
      vs = torch.rand(n, d)
      # rescale from [0, 1] to [-1, 1] so that the vectors point in more random directions
      vs = (vs * 2) - 1
      for v in vs:
        k_random = fs_random.put(v)
        k_ortho = fs_ortho.put(v)
        python_dict_random[k_random] = v
        python_dict_ortho[k_ortho] = v
      scores_random = []
      scores_ortho = []
      for k_random, v in python_dict_random.items():
        scores_random.append(F.cosine_similarity(v, fs_random.get(k_random), dim=0))
      for k_ortho, v in python_dict_ortho.items():
        scores_ortho.append(F.cosine_similarity(v, fs_ortho.get(k_ortho), dim=0))

      new_record_random.extend([np.mean(scores_random), np.std(scores_random)])
      new_record_ortho.extend([np.mean(scores_ortho), np.std(scores_ortho)])
      records_random.append(new_record_random)
      records_ortho.append(new_record_ortho)

  df_random = pd.DataFrame.from_records(records_random, columns=['d', 'n', 'cosine_sim', 'std'])
  df_ortho = pd.DataFrame.from_records(records_ortho, columns=['d', 'n', 'cosine_sim', 'std'])

  return df_random, df_ortho



def main():
  df_random, df_ortho = build_dfs()
  # Random keys
  plot_accuracy(
    df=df_random,
    title="Increasing <i>d</i> stabilizes retrieval and improves accuracy",
  )
  # Orthogonal keys
  plot_accuracy(
    df=df_ortho,
    title="Fuzzy Store can store <i>d</i> key-value pairs prefectly if it uses orthogonal keys<br><sup>...but then it explodes</sup>",
  )
  # interactive curve fit picker
  plot_curve_fits(df_random)


if __name__ == "__main__":
  main()
