
# parsetab.py
# This file is automatically generated. Do not edit.
# pylint: disable=W,C,R
_tabversion = '3.10'

_lr_method = 'LALR'

_lr_signature = 'startAUDIO EQUALS EVALUATE IDENTIFIER MODEL STRING USINGstart : statement\n             | start statementstatement : AUDIO IDENTIFIER EQUALS STRING\n                 | MODEL IDENTIFIER EQUALS STRINGstatement : EVALUATE IDENTIFIER USING IDENTIFIER'
    
_lr_action_items = {'AUDIO':([0,1,2,6,13,14,15,],[3,3,-1,-2,-3,-4,-5,]),'MODEL':([0,1,2,6,13,14,15,],[4,4,-1,-2,-3,-4,-5,]),'EVALUATE':([0,1,2,6,13,14,15,],[5,5,-1,-2,-3,-4,-5,]),'$end':([1,2,6,13,14,15,],[0,-1,-2,-3,-4,-5,]),'IDENTIFIER':([3,4,5,12,],[7,8,9,15,]),'EQUALS':([7,8,],[10,11,]),'USING':([9,],[12,]),'STRING':([10,11,],[13,14,]),}

_lr_action = {}
for _k, _v in _lr_action_items.items():
   for _x,_y in zip(_v[0],_v[1]):
      if not _x in _lr_action:  _lr_action[_x] = {}
      _lr_action[_x][_k] = _y
del _lr_action_items

_lr_goto_items = {'start':([0,],[1,]),'statement':([0,1,],[2,6,]),}

_lr_goto = {}
for _k, _v in _lr_goto_items.items():
   for _x, _y in zip(_v[0], _v[1]):
       if not _x in _lr_goto: _lr_goto[_x] = {}
       _lr_goto[_x][_k] = _y
del _lr_goto_items
_lr_productions = [
  ("S' -> start","S'",1,None,None,None),
  ('start -> statement','start',1,'p_start','dsl_v2.py',114),
  ('start -> start statement','start',2,'p_start','dsl_v2.py',115),
  ('statement -> AUDIO IDENTIFIER EQUALS STRING','statement',4,'p_statement_assign','dsl_v2.py',119),
  ('statement -> MODEL IDENTIFIER EQUALS STRING','statement',4,'p_statement_assign','dsl_v2.py',120),
  ('statement -> EVALUATE IDENTIFIER USING IDENTIFIER','statement',4,'p_statement_evaluate','dsl_v2.py',131),
]
