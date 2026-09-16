// Filename: dnaNode.h
// Created by:  shochet (28Mar00)
//
////////////////////////////////////////////////////////////////////
#ifndef DNANODE_H
#define DNANODE_H
//
////////////////////////////////////////////////////////////////////
// Includes
////////////////////////////////////////////////////////////////////
#include "pandaNode.h"
#include "dnaGroup.h"
#include "pointerTo.h"
#include "dnaStorage.h"
#include "indent.h"

class EXPCL_TOONTOWN DNANode : public DNAGroup {
PUBLISHED:
  DNANode(const std::string &initial_name);
  DNANode(const DNANode &node);

  virtual NodePath traverse(NodePath &parent, DNAStorage *store, int editing=0);
  virtual void write(std::ostream &out, DNAStorage *store, int indent_level = 0) const;

  INLINE void set_pos(const LVecBase3 &pos);
  INLINE LVecBase3 get_pos() const;

  INLINE void set_hpr(const LVecBase3 &hpr);
  INLINE LVecBase3 get_hpr() const;

  INLINE void set_scale(const LVecBase3 &scale);
  INLINE LVecBase3 get_scale() const;

protected:
  LVecBase3 _pos;
  LVecBase3 _hpr;
  LVecBase3 _scale;

private:
  virtual DNAGroup* make_copy();

public:
  static TypeHandle get_class_type() {
    return _type_handle;
  }
  static void init_type() {
    DNAGroup::init_type();
    register_type(_type_handle, "DNANode",
                  DNAGroup::get_class_type()
                  );
  }
  virtual TypeHandle get_type() const {
    return get_class_type();
  }
  virtual TypeHandle force_init_type() {init_type(); return get_class_type();}

private:
  static TypeHandle _type_handle;
};

#include "dnaNode.I"

#endif



