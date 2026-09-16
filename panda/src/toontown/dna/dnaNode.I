// Filename: dnaNode.I
// Created by:  shochet (25May00)
//
////////////////////////////////////////////////////////////////////


INLINE void DNANode::
set_pos(const LVecBase3 &pos) {
  _pos = pos;
}

INLINE LVecBase3 DNANode::
get_pos() const {
  return _pos;
}

INLINE void DNANode::
set_hpr(const LVecBase3 &hpr) {
  _hpr = hpr;
}


INLINE LVecBase3 DNANode::
get_hpr() const {
  return _hpr;
}


INLINE void DNANode::
set_scale(const LVecBase3 &scale) {
  _scale = scale;
}


INLINE LVecBase3 DNANode::
get_scale() const {
  return _scale;
}
