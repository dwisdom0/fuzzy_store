import torch

class FuzzyStore:
  def __init__(self, d: int, use_orthogonal_keys: bool=False):
    self.d = d
    self.capacity = d
    self.ortho = use_orthogonal_keys
    self.m = torch.zeros((d, d))


  def put(self, v):
    if v.shape != torch.Size([self.d]):
      raise ValueError(f'vector must be 1D with length {self.d}. {v.shape=}')

    if self.ortho:
      return self._put_ortho(v)
    return self._put_random(v)


  def _put_ortho(self, v):
    # use a random key if we've used up all of our
    # perfectly orthogonal keys
    if self.capacity <= 0:
      k = torch.rand(self.d)
      k = (k * 2) - 1
      self.m += torch.outer(v, k)
      return k

    # use a key like
    # [0, 0, 0, 1, 0, 0]
    # so it's orthogonal to the other keys we've given out so far
    k = torch.zeros(self.d)
    k[self.capacity-1] = 1.0
    self.m += torch.outer(v, k)
    self.capacity -= 1
    return k


  def _put_random(self, v):
    # construct a random key and return it to the caller so they can get their value back
    # have to do the ol' *2 -1 trick here to move them to the range [-1,1]
    # so they can point in more random directions
    # otherwise they're all in [0, 1] and point in roughly the same direction
    k = torch.rand(self.d)
    k = (k * 2) - 1
    self.m += torch.outer(v, k)
    return k


  def get(self, k):
    return self.m @ k


  def remove(self, k):
    stored_v = self.get(k)
    self.m -= torch.outer(stored_v, k)
