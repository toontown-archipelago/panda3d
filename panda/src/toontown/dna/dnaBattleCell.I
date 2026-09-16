// Filename: dnaBattleCell.I
// Created by:
//
////////////////////////////////////////////////////////////////////
INLINE void DNABattleCell::set_width_height(float width, float height) {
  _width = width;
  _height = height;
}

////////////////////////////////////////////////////////////////////
//     Function: DNABattleCell::get_width
//       Access: Public
//  Description:
////////////////////////////////////////////////////////////////////
INLINE float DNABattleCell::get_width() const {
  return _width;
}

////////////////////////////////////////////////////////////////////
//     Function: DNABattleCell::get_height
//       Access: Public
//  Description:
////////////////////////////////////////////////////////////////////
INLINE float DNABattleCell::get_height() const {
  return _height;
}


INLINE LPoint3 DNABattleCell::get_pos() const {
  return _pos;
}

INLINE void DNABattleCell::set_pos(LPoint3 pos) {
  _pos = pos;
}
