cdef extern from "string":
    ctypedef struct c_string "std::string"

cdef extern from "numberizer.hpp":
    ctypedef struct c_numberizer "numberizer<std::string>":
        int word_to_index(c_string)
        c_string index_to_word(int)
        int begin_index()
        int end_index()
    c_numberizer *new_numberizer "new numberizer<std::string>" ()
    void del_numberizer "delete" (c_numberizer *)
