# coding=utf-8

"""Miscellaneous utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pprint
import re

# Camel case to snake case utils
_first_cap_re = re.compile("(.)([A-Z][a-z0-9]+)")
_all_cap_re = re.compile("([a-z0-9])([A-Z])")


def camelcase_to_snakecase(name):
  s1 = _first_cap_re.sub(r"\1_\2", name)
  return _all_cap_re.sub(r"\1_\2", s1).lower()


def snakecase_to_camelcase(name):
  return "".join([w[0].upper() + w[1:] for w in name.split("_")])


def pprint_hparams(hparams):
  """Represents hparams using its dictionary and calls pprint.pformat on it."""
  return "\n{}".format(pprint.pformat(hparams.values(), width=1))
