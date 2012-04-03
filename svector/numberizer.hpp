#ifndef NUMBERIZER_HPP
#define NUMBERIZER_HPP

#include <vector>
#include <tr1/unordered_map>
#include <cassert>

#include <iostream> // for debug output

template<typename K>
class numberizer {
public:
  typedef int index_type;

private:
  typedef std::tr1::unordered_map<K,index_type> w2i_type;
  typedef typename std::tr1::unordered_map<K,index_type>::iterator w2i_iterator;
  typedef typename std::tr1::unordered_map<K,index_type>::const_iterator w2i_const_iterator;
  w2i_type w2i;
  std::vector<K> i2w;
  
public:
  index_type word_to_index(K const& w) {
    index_type i;
    w2i_iterator it = w2i.find(w);
    if (it == w2i.end()) {
      i = i2w.size();
      w2i[w] = i;
      i2w.push_back(w);
    } else {
      i = it->second;
    }
    return i;
  }
  K index_to_word(index_type i) const {
    assert(i < i2w.size());
    return i2w.at(i);
  }
  index_type begin_index() const { return 0; }
  index_type end_index() const { return i2w.size(); }
};

#endif

