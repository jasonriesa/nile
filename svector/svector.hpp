#ifndef SVECTOR_HPP
#define SVECTOR_HPP

#include <map>
#include <tr1/unordered_map>
#include <cassert>

template <class F, class V>
class svector {
  std::map<F,V> m;

public:
  typedef typename std::map<F,V>::size_type size_type;
  typedef typename std::map<F,V>::value_type value_type;
  typedef typename std::map<F,V>::iterator iterator;
  typedef typename std::map<F,V>::const_iterator const_iterator;

public:
  svector() { }

  svector(const F& f, const V& v) {
    m.insert(std::pair<F,V>(f, v));
  }

  // STL forwarding methods

  void swap(svector<F,V> &other) { m.swap(other.m); }
  size_type size() const { return m.size(); }
  void erase(const F& f) { m.erase(f); }
  V& operator[] (const F& f) { return m[f]; }
  iterator find(const F& f) { return m.find(f); }
  const_iterator find(const F& f) const { return m.find(f); }
  std::pair<iterator, bool> insert(const value_type& fv) { return m.insert(fv); }
  iterator begin() { return m.begin(); }
  const_iterator begin() const { return m.begin(); }
  iterator end() { return m.end(); }
  const_iterator end() const { return m.end(); }

  // Vector algebra

  svector& operator*= (const V& c) {
    for (iterator it=m.begin(); it != m.end(); ++it)
      it->second *= c;
    return *this;
  }

  svector& operator/= (const V& c) {
    for (iterator it=m.begin(); it != m.end(); ++it)
      it->second /= c;
    return *this;
  }

  svector operator- () const {
    return *this * -1.0;
  }

protected:
  template <typename C>
  svector& add (const svector<F,V>& y, const C& c) {
    // add c*y to self

    // no matter what, we have to walk through all of y

    if (y.size() < 0.1*m.size()) {

      // O(|y| log |x|)
      for (const_iterator it=y.begin(); it != y.end(); ++it) {
	(*this)[it->first] += c*it->second;
      }

    } else {

      // O(|x|+|y|)
      iterator xit = m.begin();
      const_iterator yit = y.begin();
      while (yit != y.end()) {
	if (xit == m.end() || xit->first > yit->first) {
	  m.insert(xit, std::pair<F,V>(yit->first, c*yit->second));
	  ++yit;
	} else if (xit->first == yit->first) {
	  xit->second += c*yit->second;
	  ++xit; ++yit;
	} else {
	  ++xit;
	}
      }

    }

    return *this;
  }
  
public:
  svector& operator+= (const svector<F,V>& y) {
    return this->add(y, 1);
  }

  svector& operator-= (const svector<F,V>& y) {
    return this->add(y, -1);
  }

protected:
  template <class Op>
  void intersect(const svector<F,V>& y, Op& op) const {
    // can't do bounded search with std::map, otherwise we would do double binary search
    // these thresholds have not been well optimized
    if (m.size() < 0.1*y.size()) {
      for (const_iterator xit=m.begin(); xit != m.end(); ++xit) {
	const_iterator yit = y.find(xit->first);
	if (yit != y.end())
	  op(xit->first, xit->second, yit->second);
      }
    } else if (0.1*m.size() > y.size()) {
      for (const_iterator yit=y.begin(); yit != y.end(); ++yit) {
	const_iterator xit = m.find(yit->first);
	if (xit != m.end())
	  op(yit->first, xit->second, yit->second);
      }
    } else {
      const_iterator xit = m.begin();
      const_iterator yit = y.begin();
      while (xit != m.end() && yit != y.end()) {
	if (xit->first > yit->first) {
	  ++yit;
	} else if (xit->first < yit->first) {
	  ++xit;
	} else {
	  op(xit->first, xit->second, yit->second);
	  ++xit; ++yit;
	}
      }
    }
  }

  struct dot_op {
    V result;
    dot_op() : result() { }
    void operator()(F f, V x, V y) { result += x*y; }
  };

  struct multiply_op {
    svector<F,V> result;
    void operator()(F f, V x, V y) { result[f] += x*y; }
  };

  struct divide_op {
    svector<F,V> result;
    void operator()(F f, V x, V y) { result[f] += x/y; }
  };

public:
  svector operator*(const svector<F,V>& y) const {
    multiply_op op;
    intersect(y, op);
    return op.result;
  }

  V dot(const svector<F,V>& y) const {
    dot_op op;
    intersect(y, op);
    return op.result;
  }

  svector operator/(const svector<F,V>& y) const {
    divide_op op;
    intersect(y, op);
    return op.result;
  }

  V normsquared(void) const {
    V z = V();
    for (const_iterator it=m.begin(); it != m.end(); ++it)
      z += it->second*it->second;
    return z;
  }
};

template <class F, class V>
svector<F,V> operator* (const svector<F,V>& x, const V& y) { const svector<F,V> z(x); z *= y; return z; }

template <class F, class V>
svector<F,V> operator* (const V& x, const svector<F,V>& y) { const svector<F,V> z(y); z *= x; return z; }

template <class F, class V>
svector<F,V> operator/ (const svector<F,V>& x, const V& y) { const svector<F,V> z(x); z /= y; return z; }

template <class F, class V>
svector<F,V> operator+ (const svector<F,V>& x, const svector<F,V>& y) { const svector<F,V> z(x); z += y; return z; }

template <class F, class V>
svector<F,V> operator- (const svector<F,V>& x, const svector<F,V>& y) { const svector<F,V> z(x); z -= y; return z; }

#endif
