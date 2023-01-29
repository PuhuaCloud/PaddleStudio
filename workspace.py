# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import workspace_pb2 as w
from utils import get_logger
import os
import os.path as osp
from threading import Thread
import traceback
import platform
import configparser
import time
import shutil
import copy
from dataset.utils import get_encoding


class Workspace():
    def __init__(self, workspace, dirname, logger):
        self.workspace = workspace
        self.current_version = "0.2.0"
        self.logger = logger

        # 导入工作空间
        self.load_workspace()
        self.stop_running = False
        self.sync_thread = self.sync_with_local(interval=5)

    def load_workspace(self):
        path = self.workspace.path
        newest_file = osp.join(self.workspace.path, 'workspace.newest.pb')
        bak_file = osp.join(self.workspace.path, 'workspace.bak.pb')
        flag_file = osp.join(self.workspace.path, 'pb.success')
        self.workspace.version = self.current_version
        try:
            if osp.exists(flag_file):
                with open(newest_file, 'rb') as f:
                    self.workspace.ParseFromString(f.read())
            elif osp.exists(bak_file):
                with open(bak_file, 'rb') as f:
                    self.workspace.ParseFromString(f.read())
            else:
                print("这是一个新的workspace")
        except Exception as e:
            print(traceback.format_exc())
            
        self.workspace.path = path
        self.recover_workspace()

    def recover_workspace(self):
        '''读取数据库信息'''
        if len(self.workspace.projects) > 0 or len(
                self.workspace.datasets) > 0:
            return
        projects_dir = os.path.join(self.workspace.path, 'projects')
        datasets_dir = os.path.join(self.workspace.path, 'datasets')
        if not os.path.exists(projects_dir):
            os.makedirs(projects_dir)
        if not os.path.exists(datasets_dir):
            os.makedirs(datasets_dir)

        max_project_id = 0
        max_dataset_id = 0
        max_task_id = 0
        for pd in os.listdir(projects_dir):
            try:
                if pd[0] != 'P':
                    continue
                if int(pd[1:]) > max_project_id:
                    max_project_id = int(pd[1:])
            except:
                continue
            info_pb_file = os.path.join(projects_dir, pd, 'info.pb')
            if not os.path.exists(info_pb_file):
                continue
            try:
                pj = w.Project()
                with open(info_pb_file, 'rb') as f:
                    pj.ParseFromString(f.read())
                self.workspace.projects[pd].CopyFrom(pj)
            except Exception as e:
                self.logger.info(traceback.format_exc())

            for td in os.listdir(os.path.join(projects_dir, pd)):
                try:
                    if td[0] != 'T':
                        continue
                    if int(td[1:]) > max_task_id:
                        max_task_id = int(td[1:])
                except:
                    continue
                info_pb_file = os.path.join(projects_dir, pd, td, 'info.pb')
                if not os.path.exists(info_pb_file):
                    continue
                try:
                    task = w.Task()
                    with open(info_pb_file, 'rb') as f:
                        task.ParseFromString(f.read())
                    self.workspace.tasks[td].CopyFrom(task)
                except Exception as e:
                    self.logger.info(traceback.format_exc())

        for dd in os.listdir(datasets_dir):
            try:
                if dd[0] != 'D':
                    continue
                if int(dd[1:]) > max_dataset_id:
                    max_dataset_id = int(dd[1:])
            except:
                continue
            info_pb_file = os.path.join(datasets_dir, dd, 'info.pb')
            if not os.path.exists(info_pb_file):
                continue
            try:
                ds = w.Dataset()
                with open(info_pb_file, 'rb') as f:
                    ds.ParseFromString(f.read())
                self.workspace.datasets[dd].CopyFrom(ds)
            except Exception as e:
                self.logger.info(traceback.format_exc())

        self.workspace.max_dataset_id = max_dataset_id
        self.workspace.max_project_id = max_project_id
        self.workspace.max_task_id = max_task_id

    # 每间隔interval秒，将workspace同步到本地数据库文件
    def sync_with_local(self, interval=2):
        def sync_func(s, interval_seconds=2):
            newest_file = osp.join(self.workspace.path, 'workspace.newest.pb')
            stable_file = osp.join(self.workspace.path, 'workspace.stable.pb')
            bak_file = osp.join(self.workspace.path, 'workspace.bak.pb')
            flag_file = osp.join(self.workspace.path, 'pb.success')
            while True:
                current_time = time.time()
                time_array = time.localtime(current_time)
                current_time = time.strftime("%Y-%m-%d %H:%M:%S", time_array)
                self.workspace.current_time = current_time

                if osp.exists(flag_file):
                    os.remove(flag_file)
                f = open(newest_file, mode='wb')
                f.write(s.workspace.SerializeToString())
                f.close()
                open(flag_file, 'w').close()
                if osp.exists(stable_file):
                    shutil.copyfile(stable_file, bak_file)
                shutil.copyfile(newest_file, stable_file)
                if s.stop_running:
                    break
                time.sleep(interval_seconds)

        t = Thread(target=sync_func, args=(self, interval))
        t.start()
        return t

    def check_hardware_env(self):
        # 判断是否有gpu,cpu值是否已经设置
        hasGpu = True
        try:
            '''data = {'path' : path}
            from .system import get_machine_info
            info = get_machine_info(data, self.machine_info)['info']
            if info is None:
                return
            if (info['gpu_num'] == 0 and self.sysstr == "Windows"):
                data['path'] = os.path.abspath(os.path.dirname(__file__))
                info = get_machine_info(data, self.machine_info)['info']'''
            from system import get_system_info
            info = get_system_info()['info']
            hasGpu = (info['gpu_num'] > 0)
            self.machine_info = info
            #driver_ver = info['driver_version']
            # driver_ver_list = driver_ver.split(".")
            # major_ver, minor_ver = driver_ver_list[0:2]
            # if sysstr == "Windows":
            #     if int(major_ver) < 411 or \
            #             (int(major_ver) == 411 and int(minor_ver) < 31):
            #         raise Exception("The GPU dirver version should be larger than 411.31")
            #
            # elif sysstr == "Linux":
            #     if int(major_ver) < 410 or \
            #             (int(major_ver) == 410 and int(minor_ver) < 48):
            #         raise Exception("The GPU dirver version should be larger than 410.48")

        except Exception as e:
            hasGpu = False

        self.m_HasGpu = hasGpu


def init_workspace(workspace, dirname, logger):
    '''初始化工作空间'''
    wp = Workspace(workspace, dirname, logger)
    return {'status': 1}


def set_attr(data, workspace):
    """对workspace中项目，数据，任务变量进行修改赋值

    Args:
        data为dict,key包括
        'struct'结构类型，可以是'dataset', 'project'或'task';
        'id'查询id, 其余的key:value则分别为待修改的变量名和相应的修改值。
    """
    struct = data['struct']
    id = data['id']
    assert struct in ['dataset', 'project', 'task'
                      ], "struct只能为dataset, project或task"
    if struct == 'dataset':
        assert id in workspace.datasets, "数据集ID'{}'不存在".format(id)
        modify_struct = workspace.datasets[id]
    elif struct == 'project':
        assert id in workspace.projects, "项目ID'{}'不存在".format(id)
        modify_struct = workspace.projects[id]
    elif struct == 'task':
        assert id in workspace.tasks, "任务ID'{}'不存在".format(id)
        modify_struct = workspace.tasks[id]
    '''for k, v in data.items():
        if k in ['id', 'struct']:
            continue
        assert hasattr(modify_struct,
                        k), "{}不存在成员变量'{}'".format(type(modify_struct), k)
        setattr(modify_struct, k, v)'''
    for k, v in data['attr_dict'].items():
        assert hasattr(modify_struct,
                       k), "{}不存在成员变量'{}'".format(type(modify_struct), k)
        setattr(modify_struct, k, v)
    with open(os.path.join(modify_struct.path, 'info.pb'), 'wb') as f:
        f.write(modify_struct.SerializeToString())

    return {'status': 1}


def get_attr(data, workspace):
    """取出workspace中项目，数据，任务变量值

    Args:
        data为dict,key包括
        'struct'结构类型，可以是'dataset', 'project'或'task';
        'id'查询id, 'attr_list'需要获取的属性值列表
    """
    struct = data['struct']
    id = data['id']
    assert struct in ['dataset', 'project', 'task'
                      ], "struct只能为dataset, project或task"
    if struct == 'dataset':
        assert id in workspace.datasets, "数据集ID'{}'不存在".format(id)
        modify_struct = workspace.datasets[id]
    elif struct == 'project':
        assert id in workspace.projects, "项目ID'{}'不存在".format(id)
        modify_struct = workspace.projects[id]
    elif struct == 'task':
        assert id in workspace.tasks, "任务ID'{}'不存在".format(id)
        modify_struct = workspace.tasks[id]

    attr = {}
    for k in data['attr_list']:
        if k in ['id', 'struct']:
            continue
        assert hasattr(modify_struct,
                       k), "{}不存在成员变量'{}'".format(type(modify_struct), k)
        v = getattr(modify_struct, k)
        attr[k] = v

    return {'status': 1, 'attr': attr}
