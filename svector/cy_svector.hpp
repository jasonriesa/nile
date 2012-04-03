#include <string>
#include <sstream>
#include <cstdlib>

#include "svector.hpp"
#include "numberizer.hpp"

static numberizer<std::string> feature_numberizer;

template <class F, class V>
void svector_setitem(svector<F,V>& v, F f, V x) { v[f] = x; }

template <class F, class V>
V svector_getitem(const svector<F,V>& v, F f) { 
  typename svector<F,V>::const_iterator it = v.find(f);
  if (it != v.end())
    return it->second;
  else {
    return V();
  }
}

template <class F, class V>
bool svector_contains(const svector<F,V>& v, F f) { 
  typename svector<F,V>::const_iterator it = v.find(f);
  return it != v.end();
}

template <class F, class V>
std::string svector_str(const svector<F,V> &sv) {
  std::ostringstream out;

  bool first = true;
  for (typename svector<F,V>::const_iterator it=sv.begin(); it != sv.end(); ++it) {
    if (!first)
      out << " ";
    out << feature_numberizer.index_to_word(it->first) << "=" << it->second;
    first = false;
  }
  return out.str();
}

template <class F, class V>
svector<F,V> *svector_from_str(const std::string &s) {
  svector<F,V> *sv = new svector<F,V>();

  std::istringstream tokenizer(s);
  std::string tok;
  int i;
  
  while (tokenizer >> tok) {
    i = tok.rfind('=');
    numberizer<std::string>::index_type feature = feature_numberizer.word_to_index(tok.substr(0, i));
    double value = std::strtod(tok.substr(i+1).c_str(), NULL);
    sv->insert(std::pair<F,V>(feature, value));
  }
  return sv;
}

template <class F, class V>
struct svector_iterator {
  typename svector<F,V>::const_iterator cur, end;
  svector_iterator(const svector<F,V>& v) : cur(v.begin()), end(v.end()) { }
  bool has_next() { return cur != end; }
  std::string key() { return feature_numberizer.index_to_word(cur->first); }
  V value() { return cur->second; }
  void next() { ++cur; }
};

