#!/usr/bin/python
import os 
import sys
import paramiko
import traceback 
import yaml

class ConfigNetwork(object):
    """This class is used to 
    """
    def __init__(self, cur_net, cur_config, 
                 mode):
               
        self.cur_config=cur_config[8:]
        self.network=cur_net
        with open("config.yaml") as fp:
            cfg=yaml.load(fp)
        if mode == "online":
            # host
            self.host=cfg["config"]["online"]["remote"]["host"]
            # username
            self.user=cfg["config"]["online"]["remote"]["user"]
            # password
            self.passwd=cfg["config"]["online"]["remote"]["pass"]
            # keyfile
            self.keyfile=cfg["config"]["online"]["remote"]["keyfile"]
            # remote code directory
            self.remote_code_dir=cfg["config"]["online"]["remote"]["network"]["code_dir"]
            self.remote_code_dir=self.remote_code_dir.replace("network",cur_net)
            # remote model directory
            self.remote_model_dir=cfg["config"]["online"]["remote"]["network"]["model_dir"]
            self.remote_model_dir=self.remote_model_dir.replace("network",cur_net)
            # remote current configuration directory
            self.remote_conf_dir=cfg["config"]["online"]["remote"]["network"]["conf_dir"]
            self.remote_conf_dir=self.remote_conf_dir.replace("network",cur_net)
            # local model directory
            self.local_model_dir=cfg["config"]["online"]["local"]["model_dir"]
            # local model directory
            self.local_conf_dir=cfg["config"]["online"]["local"]["conf_dir"]
            # current configuration
            self.store_conf_online()
            self.get_model_online()
        elif mode=="offline":
            self.get_model_offline()
        else:
            print ("[ERROR]: mode not supported")
            
    def get_model_online(self):
        """This is a function for establishing remote connection
        """
        try:
            key=paramiko.RSAKey.from_private_key_file(self.keyfile)
            ssh_client=paramiko.SSHClient()
            ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh_client.connect(hostname=self.host, username=self.user, password=self.passwd, 
                               pkey=key)
            ftp_client=ssh_client.open_sftp()
            ftp_client.put(self.local_conf_dir,self.remote_conf_dir)
            command= "sudo python3 "+str(self.remote_code_dir)
            stdin, stdout, stderr=ssh_client.exec_command(command)
            print (stdout.readlines())
            ftp_client.get(self.remote_model_dir, self.local_model_dir)
            ftp_client.close()
        except:
            traceback.print_exc()
            print ("[ERROR]: could not ssh")
    
    def get_model_offline(self):
        """This is a function for establishing remote connection
        """
        try:
            print ("cur_net")
        except:
            traceback.print_exc()
            print ("[ERROR]: could not ssh")
    
    def store_conf_online(self):
        """This function is used to store current configuration to yaml
        """
        conf=dict(cur_conf=self.cur_config)
        with open ("cur_config.yaml","w", ) as curfp:
            yaml.dump(conf, curfp, default_flow_style=False)



