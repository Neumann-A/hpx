//  Copyright (c) 2022 Alexander Neumann
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/batch_environments/hpcpack_environment.hpp>
#include <hpx/util/from_string.hpp>

#include <algorithm>
#include <boost/tokenizer.hpp>
#include <boost/algorithm/string/replace.hpp>

#include <cstddef>
#include <string>
#include <vector>
#include <iostream>

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
        auto nodes_core_list = std::getenv("CCP_NODES");
        if (valid_ = (nodes_core_list != nullptr); valid_)
        {

            const std::string nodes_str{nodes_core_list};

            // Initialize our node number
            boost::char_separator<char> sep(" ");
            boost::tokenizer<boost::char_separator<char>> tok(nodes_str, sep);

            // Number of Nodes is first element
            const std::string num_localities_str{*std::begin(tok)};
            num_localities_ =
                from_string<std::size_t>(num_localities_str, std::size_t(1));



            auto current_node_name = std::getenv("COMPUTERNAME");
            const std::string current_node_str{current_node_name};
            // Calculate node number from CCP_NODES by finding COMPUTERNAME
            auto found_node_pos =
                std::find(std::begin(tok), std::end(tok), current_node_str);
            if(found_node_pos == std::end(tok)) {
                // Currrent COMPUTERNAME not in CCP_NODES -> invalid environment
                valid_ = false;
                return;
            }
            node_num_ = ((static_cast<std::size_t>(std::distance(std::begin(tok),found_node_pos)) - 1) / 2); 

            // -1 Since first element ist number of nodes; div 2 due to CCP_NODES being pairs of <name> <avail_cores>
            if (have_mpi)
            {
                // Initialize our node number from MPI env. 
                char* var = std::getenv("PMI_RANK");
                if (var != nullptr)
                {
                    node_num_ = from_string<std::size_t>(var);
                }
            }

            // Get assigned core numbers and create a hpc binding. 
            char* core_list = std::getenv("CCP_COREIDS");
            if (core_list != nullptr)
            {
                const std::string core_list_str{core_list};
                boost::tokenizer<boost::char_separator<char>> tok(
                    core_list_str, sep);
                num_threads_ = static_cast<std::size_t>(
                    std::distance(std::begin(tok), std::end(tok)));
                core_bind_ = std::string{"thread:all=pu:"};
                core_bind_.append(core_list);
                boost::algorithm::replace_all(core_bind_, " ", ",");
            }
            else
            {
                const std::string num_cores_str{*(++found_node_pos)};
                num_threads_ = from_string<std::size_t>(num_cores_str);
            }

            // Populate nodelist.
            if (nodelist.empty())
            {
                auto iter = tok.begin();
                for(++iter; iter != tok.end(); ++(++iter)) {
                    // Move one since 1. element is the number of nodes
                    // Move two since the list is always <node> <nr_of_cores>
                    nodelist.emplace_back(*iter);
                }
            }

            if(debug) {
                std::cerr << "HPC Pack nodelist: " 
                          << nodes_str << std::endl;
                std::cerr << "Localities: " 
                          << num_localities_ << std::endl;
                std::cerr << "Threads: " 
                          << num_threads_ << std::endl;
                std::cerr << "Name: " 
                          << current_node_name << std::endl;
            }
        }
    }
}}}
