//  Copyright (c) 2022 Alexander Neumann
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/batch_environments/hpcpack_environment.hpp>
#include <hpx/string_util/classification.hpp>
#include <hpx/string_util/split.hpp>
#include <hpx/util/from_string.hpp>

#include <algorithm>
#include <atomic>
#include <boost/tokenizer.hpp>
#include <boost/algorithm/string/replace.hpp>

#include <cstddef>
#include <iostream>
#include <string_view>
#include <string>
#include <charconv>
#include <utility>
#include <vector>

// Env variables for HPC Pack can be found at:
// https://docs.microsoft.com/en-us/previous-versions/windows/it-pro/windows-hpc-server-2008r2/gg286970(v=ws.10)

// COMPUTERNAME -> 

//CCP_TASKID=2
//CCP_TASKCONTEXT=15317.26014
//CCP_CLUSTER_NAME=HEAD
//CCP_JOBID=15317
//CCP_DATA=C:\Program Files\Microsoft HPC Pack 2016\Data\
//CCP_JOBNAME=set test
//CCP_NODES_CORES=2 NODE02 80 NODE03 80
//CCP_JOBTYPE=Batch
//CCP_RETRY_COUNT=0
//CCP_HOME=C:\Program Files\Microsoft HPC Pack 2016\
//CCP_JOBTEMPLATE=Default
//CCP_MPI_WORKDIR=%SCRATCH_DIR%\neumann\bin
//CCP_NUMCPUS=160
//CCP_RERUNNABLE=False
//CCP_LOGROOT_USR=%LOCALAPPDATA%\Microsoft\Hpc\LogFiles\
//CCP_TASKINSTANCEID=0
//CCP_PREVIOUS_JOB_ID=14868
//CCP_SCHEDULER=Head
//CCP_STDOUT=C:\Scratch\neumann\logs\out_15317_2.txt
//CCP_NODES=2 NODE02 80 NODE03 80
//CCP_SERVICEREGISTRATION_PATH=CCP_REGISTRATION_STORE;\\HEAD\HpcServiceRegistration
//CCP_SERVICEREGISTRATION_PATH=CCP_REGISTRATION_STORE;\\HEAD\HpcServiceRegistration
//CCP_TASKSYSTEMID=26014
//CCP_OWNER_SID=S-1-5-21-623046577-3442102450-3314793965-5699
//CCP_WORKDIR=C:\Scratch\neumann\bin
//CCP_STDERR=C:\Scratch\neumann\logs\out_15317_2.txt
//CCP_CONNECTIONSTRING=Head
//CCP_EXCLUSIVE=False
//CCP_LOGROOT_SYS=C:\Program Files\Microsoft HPC Pack 2016\Data\LogFiles\
//CCP_ENVLIST=CCP_TASKSYSTEMID,HPC_RUNTIMESHARE,CCP_TASKINSTANCEID,CCP_JOBID,CCP_TASKID
//CCP_RUNTIME=2147483647
//CCP_MPI_NETMASK=141.83.112.0/255.255.255.0
//CCP_COREIDS=0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79

namespace hpx { namespace util { namespace batch_environments { 
    hpcpack_environment::hpcpack_environment(
        std::vector<std::string>& nodelist, bool have_mpi, bool debug)
      : node_num_(std::size_t(-1))
      , num_localities_(std::size_t(-1))
      , num_threads_(std::size_t(-1))
      , valid_(false)
    {
        char* node_num = std::getenv("CCP_TASKSYSTEMID"); //Not certain about this one. 

        valid_ = node_num != nullptr;
        if (valid_)
        {
            // Initialize our node number
            node_num_ = from_string<std::size_t>(node_num, std::size_t(1));
            auto cpp_nodes = std::getenv("CCP_NODES");
            auto num_localities = std::string(cpp_nodes);
            {
                boost::char_separator<char> sep(" ");
                boost::tokenizer<boost::char_separator<char>> tok(
                       num_localities, sep);
                const std::string number_str{*tok.begin()}; 
                num_localities_ = from_string<std::size_t>(number_str, std::size_t(1));
            }

            if (have_mpi)
            {
                // Initialize our node number, if available
                char* var = std::getenv("PMI_RANK");
                if (var != nullptr)
                {
                    node_num_ = from_string<std::size_t>(var);
                }
            }
            else if (num_localities_ > 1)
            {
                valid_ = false;
            }

            // Get the number of threads, if available
            char* var = std::getenv("CCP_COREIDS");
            if (var != nullptr)
            {
                boost::char_separator<char> sep(" ");
                boost::tokenizer<boost::char_separator<char>> tok(
                    std::string(var), sep);
                num_threads_ = static_cast<std::size_t>(
                    std::distance(std::begin(tok), std::end(tok))) + 1;
            }
            else if ((var = std::getenv("CCP_NODES")) != nullptr)
            {
                boost::char_separator<char> sep(" ");
                boost::tokenizer<boost::char_separator<char>> tok(
                    std::string(var), sep);
                auto node_name = std::getenv("COMPUTERNAME"); // Should always be defined 
                auto found_node_pos = std::find(std::begin(tok),std::end(tok),std::string(node_name));
                num_threads_ =  from_string<std::size_t>(*(found_node_pos++));
            }
            else
            {
                valid_ = false;
            }

            if (nodelist.empty())
            {
                read_node_env(nodelist, have_mpi, debug);
            }
            char * core_list = std::getenv("CCP_COREIDS");
            if(core_list != nullptr) {
                core_bind_ = std::string{"thread:all=core:"};
                core_bind_.append(core_list);
                boost::algorithm::replace_all(core_bind_, " ", ",");
            }
        }
    }
    void hpcpack_environment::read_node_env(std::vector<std::string>& nodelist, bool have_mpi, bool debug) {
        char * nodelist_env = std::getenv("CCP_NODES");
        boost::char_separator<char> sep(" ");
        boost::tokenizer<boost::char_separator<char>> tok(
                    std::string(nodelist_env), sep);
        auto iter = tok.begin();
        for(++iter; iter != tok.end(); ++(++iter)) {
            // Move one since 1. element is the number of nodes
            // Move two since the list is always <node> <nr_of_cores>
            nodelist.emplace_back(*iter);
        }
    }
}}}
