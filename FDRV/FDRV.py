'''
Filename: c:\Users\alexf\OneDrive\Desktop\projetofinalcurso\FDRV\FDRV.py
Path: c:\Users\alexf\OneDrive\Desktop\projetofinalcurso\FDRV
Created Date: Saturday, November 3rd 2018, 11:53:27 am
Author: Alexandre Frazao

Copyright (c) 2018  
'''
import jarray
import inspect
import os
from java.lang import System
from java.util.logging import Level
from org.sleuthkit.datamodel import SleuthkitCase
from org.sleuthkit.datamodel import AbstractFile
from org.sleuthkit.datamodel import ReadContentInputStream
from org.sleuthkit.datamodel import BlackboardArtifact
from org.sleuthkit.datamodel import BlackboardAttribute
from org.sleuthkit.autopsy.ingest import IngestModule
from org.sleuthkit.autopsy.ingest.IngestModule import IngestModuleException
from org.sleuthkit.autopsy.ingest import DataSourceIngestModule
from org.sleuthkit.autopsy.ingest import FileIngestModule
from org.sleuthkit.autopsy.ingest import IngestModuleFactoryAdapter
from org.sleuthkit.autopsy.ingest import IngestMessage
from org.sleuthkit.autopsy.ingest import IngestServices
from org.sleuthkit.autopsy.coreutils import Logger
from org.sleuthkit.autopsy.casemodule import Case
from org.sleuthkit.autopsy.casemodule.services import Services
from org.sleuthkit.autopsy.casemodule.services import FileManager
from org.sleuthkit.autopsy.casemodule.services import Blackboard

from java.awt import BorderLayout, GridLayout, FlowLayout, Dimension
from java.awt.event import KeyAdapter, KeyEvent, KeyListener
from threading import Thread


#   ___ _     _          _                __ _                    _   _             
#  / __| |___| |__  __ _| |  __ ___ _ _  / _(_)__ _ _  _ _ _ __ _| |_(_)___ _ _  ___
# | (_ | / _ \ '_ \/ _` | | / _/ _ \ ' \|  _| / _` | || | '_/ _` |  _| / _ \ ' \(_-<
#  \___|_\___/_.__/\__,_|_| \__\___/_||_|_| |_\__, |\_,_|_| \__,_|\__|_\___/_||_/__/
#                                             |___/                                 

#
# Executable options
#

# TODO: Try to find Autopsy allowed threads
NUMBER_THREADS = 4
NUMBER_FRAMES_TO_SKIP = 5
ALLOWED_EXTENSIONS = [".mp4", ".avi"]
GENERATE_RESULT_AS_VIDEO = True

#
# Module outputs
#

# Name of file to hold the filenames where faces were detected
C_FACES_FOUND_FNAME = "FDRV_faces_found.txt"

# Name of file to hold the files where recognition occurred 
C_FDRI_WANTED_FNAME = "FDRV_wanted.txt"

# Name of created DFXML file
C_DFXML_FNAME = "dfxml.xml"

# Name of file to register filenames and size
C_FILE_WITH_FNAMES_AND_SIZES = "FDRV_filenames+size.log.txt"

# Name of file to get the list of repeated files
C_REPEATED_FILES_LOG = "FDRV_repeated_files.log.txt"

# Name of file holding JSON parameters
C_PARAMS_JSON_FNAME="params.json"

#  __  __         _      _          _                 _      __ _      _ _   _          
# |  \/  |___  __| |_  _| |___   __| |__ _ ______  __| |___ / _(_)_ _ (_) |_(_)___ _ _  
# | |\/| / _ \/ _` | || | / -_) / _| / _` (_-<_-< / _` / -_)  _| | ' \| |  _| / _ \ ' \ 
# |_|  |_\___/\__,_|\_,_|_\___| \__|_\__,_/__/__/ \__,_\___|_| |_|_||_|_|\__|_\___/_||_|
#                                                                                       

# Factory that defines the name and details of the module and allows Autopsy
# to create instances of the modules that will do the analysis.
class FDRVModuleFactory(IngestModuleFactoryAdapter):

    moduleName = "Facial Detection and Recognition in Videos"
    moduleVersion = "V-dev1.0"

    def getModuleDisplayName(self):
        return self.moduleName

    def getModuleDescription(self):
        return "Sample module that does X, Y, and Z."

    def getModuleVersionNumber(self):
        return self.moduleVersion

    def isDataSourceIngestModuleFactory(self):
        return True

    def createDataSourceIngestModule(self, ingestOptions):
        return FDRIVModule(self.settings)

    def hasIngestJobSettingsPanel(self):
        return True

    def getDefaultIngestJobSettings(self):
        return UISettings()

    def getIngestJobSettingsPanel(self, settings):
        self.settings = settings
        return UISettingsPanel(self.settings)


# Data Source-level ingest module.  One gets created per data source.
class FDRIVModule(DataSourceIngestModule):

    _logger = Logger.getLogger(FDRVModuleFactory.moduleName)

    def log(self, level, msg):
        self._logger.logp(level, self.__class__.__name__, inspect.stack()[1][3], msg)

    def __init__(self):
        self.context = None
        self.localSettings = settings
        self.models = {
            "recognition"   : "",
            "detection"     : "",
            "shape"         : ""
        }

    # 'context' is an instance of org.sleuthkit.autopsy.ingest.IngestJobContext.
    # See: http://sleuthkit.org/autopsy/docs/api-docs/4.6.0/classorg_1_1sleuthkit_1_1autopsy_1_1ingest_1_1_ingest_job_context.html
    def startUp(self, context):
        
        # Throw an IngestModule.IngestModuleException exception if there was a problem setting up
        # raise IngestModuleException("Oh No!")
        self.context = context

        #
        # File verification
        #
        
        if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "FDRV.exe")):
            self.log(Level.ERROR, "FDRV Executable not found! Terminating")

        # TODO: Provide these as options for user?

        self.models["recognition"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dlib_face_recognition_resnet_model_v1.dat")
        if not os.path.exists(self.models["recognition"]):
            self.log(Level.ERROR, "Recognition model not found! Terminating")
        
        self.models["detection"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mmod_human_face_detector.dat")
        if not os.path.exists(self.models["detection"]):
            self.log(Level.ERROR, "face detection model not found! Terminating")
        
        self.models["shape"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shape_predictor_5_face_landmarks.dat")
        if not os.path.exists(self.models["shape"]):
            self.log(Level.ERROR, "Shape predictor model not found! Terminating")
        

    #   ___ _        _   ___                  
    #  / __| |_ _  _| |_|   \ _____ __ ___ _  
    #  \__ \ ' \ || |  _| |) / _ \ V  V / ' \ 
    #  |___/_||_\_,_|\__|___/\___/\_/\_/|_||_|

    def shutDown(self):
        """shutdown code"""
        # TODO: File cleanup maybe?
        # TODO: Run report?
        # TODO: Check if errors or log meta information?

    #  __  __      _        _           _    
    # |  \/  |__ _(_)_ _   | |___  __ _(_)__ 
    # | |\/| / _` | | ' \  | / _ \/ _` | / _|
    # |_|  |_\__,_|_|_||_| |_\___/\__, |_\__|
    #                             |___/      

    # The 'dataSource' object being passed in is of type org.sleuthkit.datamodel.Content.
    # See: http://www.sleuthkit.org/sleuthkit/docs/jni-docs/4.6.0/interfaceorg_1_1sleuthkit_1_1datamodel_1_1_content.html
    # 'progressBar' is of type org.sleuthkit.autopsy.ingest.DataSourceIngestModuleProgress
    # See: http://sleuthkit.org/autopsy/docs/api-docs/4.6.0/classorg_1_1sleuthkit_1_1autopsy_1_1ingest_1_1_data_source_ingest_module_progress.html
    def process(self, dataSource, progressBar):

        # we don't know how much work there is yet
        progressBar.switchToIndeterminate()

        # Blackboard API:  http://sleuthkit.org/autopsy/docs/api-docs/4.6.0/classorg_1_1sleuthkit_1_1autopsy_1_1casemodule_1_1services_1_1_blackboard.html
        blackboard = Case.getCurrentCase().getServices().getBlackboard()

        # FileManager API: http://sleuthkit.org/autopsy/docs/api-docs/4.6.0/classorg_1_1sleuthkit_1_1autopsy_1_1casemodule_1_1services_1_1_file_manager.html
        fileManager = Case.getCurrentCase().getServices().getFileManager()
        
        # Querying all videos
        files = []
        for extension in ALLOWED_EXTENSIONS:
            try:
                files.extend(fileManager.findFiles(dataSource, "%" + extension))
            except TskCoreException:
                self.log(Level.INFO, "Error getting files from: '" + extension + "'")

        numFiles = len(files)
        if len(numFiles) == 0:
            self.log(Level.ERROR, "Didn't find any usable files! Terminating")
            return DataSourceIngestModule.ProcessResult.OK
    
        self.log(Level.INFO, "Found " + str(numFiles) + " files")
        
        module_output_dir = Case.getCurrentCase().getModuleDirectory()
        module_dir = os.path.join(output_dir,dataSource.getName(),C_FDRI_DIR)

        # Calling thread to do the work
        # This can/will block for a long time
        executable_thread = Thread(target=lambda: self.thread_work(self.pathToExe, configFilePath))
        executable_thread.start()

        # Killing thread if user press cancel
        # Seems kinda bad but it's the most responsive way possible to cancel the process
        while(executable_thread.isAlive()):
            # Checking cancel every secund
            if self.context.isJobCancelled():
                self.log(Level.INFO, "User cancelled job! Terminating thread")
                JThread.interrupt(executable_thread)
                self.log(Level.INFO, "Thread terminated")
                self.deleteFiles(module_dir)
                return IngestModule.ProcessResult.OK
            time.sleep(1)



        #Post a message to the ingest messages in box.
        message = IngestMessage.createMessage(IngestMessage.MessageType.DATA,
            "Sample Jython Data Source Ingest Module", "Found %d files" % fileCount)
        IngestServices.getInstance().postMessage(message)

        return IngestModule.ProcessResult.OK

    #   _  _     _                  __              _   _             
    #  | || |___| |_ __  ___ _ _   / _|_  _ _ _  __| |_(_)___ _ _  ___
    #  | __ / -_) | '_ \/ -_) '_| |  _| || | ' \/ _|  _| / _ \ ' \(_-<
    #  |_||_\___|_| .__/\___|_|   |_|  \_,_|_||_\__|\__|_\___/_||_/__/
    #             |_|                                                 
    
    # File cleanup
    def deleteFiles(self, path):
        # ignoring the error if the directory is empty
        shutil.rmtree(path, ignore_errors=True)

    # Subprocess initiator
    def thread_work(self, path, param_path, min_size=0, max_size=0):

        sub_args = [path, "--params", param_path]
        if min_size > 0:
            sub_args.extend(["--min", str(min_size)])

        if max_size > 0:
            sub_args.extend(["--max", str(max_size)])

        returnCode = subprocess.call(sub_args)
        if returnCode:
            Err_S = "Error in executable: got '%s'" % (str(returnCode))
            self.log(Level.SEVERE,Err_S)
            if returnCode <= len(self.errorList) and returnCode > 0:
                self.log(Level.SEVERE, self.errorList[returnCode])
        else:
            msg_S = "Child process FDRI.exe terminated with no problems"
            self.log(Level.INFO, msg_S)

#    ___                _             _           _   _   _                    _     _        _        _            
#   / __|__ _ ___ ___  | |_____ _____| |  ___ ___| |_| |_(_)_ _  __ _ ___  ___| |__ (_)___ __| |_   __| |__ _ ______
#  | (__/ _` (_-</ -_) | / -_) V / -_) | (_-</ -_)  _|  _| | ' \/ _` (_-< / _ \ '_ \| / -_) _|  _| / _| / _` (_-<_-<
#   \___\__,_/__/\___| |_\___|\_/\___|_| /__/\___|\__|\__|_|_||_\__, /__/ \___/_.__// \___\__|\__| \__|_\__,_/__/__/
#                                                               |___/             |__/                              
class UISettings(IngestModuleIngestJobSettings):
    serialVersionUID = 1L

    def __init__(self):
        placeholder = False

#    ___                _             _           _   _   _                _   _ ___      _            
#   / __|__ _ ___ ___  | |_____ _____| |  ___ ___| |_| |_(_)_ _  __ _ ___ | | | |_ _|  __| |__ _ ______
#  | (__/ _` (_-</ -_) | / -_) V / -_) | (_-</ -_)  _|  _| | ' \/ _` (_-< | |_| || |  / _| / _` (_-<_-<
#   \___\__,_/__/\___| |_\___|\_/\___|_| /__/\___|\__|\__|_|_||_\__, /__/  \___/|___| \__|_\__,_/__/__/
#                                                               |___/                                  
class UISettingsPanel(IngestModuleIngestJobSettingsPanel):

    def __init__(self, settings):
        self.localSettings = settings
      
    def getSettings(self):
        return self.localSettings



