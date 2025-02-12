//  tab_check implementation  ------------------------------------------------//

//  Copyright Beman Dawes 2002.
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#include <boost/foreach.hpp>
#include <boost/tokenizer.hpp>
#include <string>
#include "function_hyper.hpp"
#include "tab_check.hpp"

namespace boost { namespace inspect {
    tab_check::tab_check()
      : m_files_with_errors(0)
    {
        register_signature(".c");
        register_signature(".cpp");
        register_signature(".cu");
        register_signature(".cxx");
        register_signature(".h");
        register_signature(".hpp");
        register_signature(".hxx");
        register_signature(".ipp");
        register_signature("Jamfile");
        register_signature(".py");
    }

    void tab_check::inspect(const string& library_name,
        const path& full_path,     // example: c:/foo/boost/filesystem/path.hpp
        const string& contents)    // contents of file to be inspected
    {
        if (contents.find("hpxinspect:"
                          "notab") != string::npos)
            return;
        string total, linenum;
        long errors = 0, currline = 0;
        size_t p = 0;
        std::vector<std::string> someline, lineorder;

        char_separator<char> sep("\n", "", boost::keep_empty_tokens);
        tokenizer<char_separator<char>> tokens(contents, sep);
        for (const auto& t : tokens)
        {
            size_t rend = t.find_first_of("\r"), size = t.size();
            if (rend == size - 1)
            {
                someline.push_back(t.substr(0, t.size() - 1));
            }
            else
            {
                char_separator<char> sep2("\r", "", boost::keep_empty_tokens);
                tokenizer<char_separator<char>> tokens2(t, sep2);
                for (const auto& u : tokens2)
                {
                    if (!u.empty() && u.back() == '\r')
                        someline.push_back(u.substr(0, u.size() - 1));
                    else
                        someline.push_back(u);
                }
            }
        }
        while (p < someline.size())
        {
            currline++;
            if (someline[p].find('\t') != string::npos)
            {
                errors++;
                linenum = std::to_string(currline);
                lineorder.push_back(linenum);
            }
            p++;
        }
        p = 0;
        while (p < lineorder.size())
        {
            total += linelink(full_path,
                lineorder[p]);    //linelink is located in function_hyper.hpp
            if (p < lineorder.size() - 1)
            {
                total += ", ";
            }
            p++;
        }
        if (errors > 0)
        {
            string errored = name() + total;
            error(library_name, full_path, errored);
            ++m_files_with_errors;
        }
    }
}}    // namespace boost::inspect
