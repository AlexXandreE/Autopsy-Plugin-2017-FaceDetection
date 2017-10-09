# -*- coding: utf-8 -*-
#
# Date: 02 October 2017
# Author: Alexandre Frazao Rosario
#         Patricio Domingues
#
# Module Full Description: 
# This is a deep neural network face detection module with facial recognition capability. 
# The module search for all the images with the selected extension and makes a copy 
# to the /temp case folder, it calls an external executable and waits for it to finish, 
# then it parses the output file and marks all files that have been targeted as faces by the executable.
# The executable is a implementation of facial detection and recognition with Dlib DNN(http://dlib.net/).
# 
# The facial recognition element is activated when selecting a folder with images from the person that 
# the program should look for, it will look for the person and if it finds, marks it as interesting file hit.
#
# All the detectors used can be found at: https://github.com/davisking/dlib-models
#
# USE INSTRUCTIONS:
#   -> Check which file extensions will be used.
#   -> Choose 1 folder with just images from the person you want to find
#       - if the folder was not provided will only do facial detection
#       - the folder must only contain photos and must be of the 3 file types: .jpg or .jpeg or .png
#   
#   NOTE: The .exe expect the natural flow of the file, 
#   if the execution is halted halfway its recommended to clean the /temp project folder
#
# See http://sleuthkit.org/autopsy/docs/api-docs/4.4/index.html for documentation

#Jython librarys
import jarray
import inspect
import os #file checking
import subprocess #.exe calling
import shutil # file copy

#Java librarys
from java.io import File
from java.lang import System
from java.util.logging import Level

#UI librarys
from javax.swing import JCheckBox
from javax.swing import JLabel
from javax.swing import BoxLayout
from java.awt import GridLayout
from java.awt import BorderLayout
from javax.swing import BorderFactory
from javax.swing import JToolBar
from javax.swing import JPanel
from javax.swing import JFrame
from javax.swing import JScrollPane
from javax.swing import JComponent
from java.awt.event import KeyListener
from java.awt.event import KeyEvent
from java.awt.event import KeyAdapter
from javax.swing.event import DocumentEvent
from javax.swing.event import DocumentListener
from javax.swing import JFileChooser
from javax.swing import JButton

#sleuthkit librarys
from org.sleuthkit.autopsy.ingest import IngestModuleIngestJobSettings
from org.sleuthkit.autopsy.ingest import IngestModuleIngestJobSettingsPanel
from org.sleuthkit.datamodel import SleuthkitCase
from org.sleuthkit.datamodel import AbstractFile
from org.sleuthkit.datamodel import ReadContentInputStream
from org.sleuthkit.datamodel import BlackboardArtifact
from org.sleuthkit.datamodel import BlackboardAttribute
from org.sleuthkit.autopsy.ingest import IngestModule
from org.sleuthkit.autopsy.ingest.IngestModule import IngestModuleException
from org.sleuthkit.autopsy.ingest import DataSourceIngestModule
from org.sleuthkit.autopsy.ingest import FileIngestModule
from org.sleuthkit.autopsy.datamodel import ContentUtils
from org.sleuthkit.autopsy.ingest import IngestModuleFactoryAdapter
from org.sleuthkit.autopsy.ingest import IngestMessage
from org.sleuthkit.autopsy.ingest import IngestServices
from org.sleuthkit.autopsy.coreutils import Logger
from org.sleuthkit.autopsy.casemodule import Case
from org.sleuthkit.autopsy.casemodule.services import Services
from org.sleuthkit.autopsy.casemodule.services import FileManager
from org.sleuthkit.autopsy.casemodule.services import Blackboard

# Factory that defines the name and details of the module and allows Autopsy
# to create instances of the modules that will do the analysis.
class FaceModuleFactory(IngestModuleFactoryAdapter):

    moduleName = "Face Detector"

    def getModuleDisplayName(self):
        return self.moduleName

    def getModuleDescription(self):
        return "Facial detection for easier image spliting"

    def getModuleVersionNumber(self):
        return "V.1.0"

    def isDataSourceIngestModuleFactory(self):
        return True

    def createDataSourceIngestModule(self, ingestOptions):
        return FaceModule(self.settings)

    def hasIngestJobSettingsPanel(self):
        return True

    def getDefaultIngestJobSettings(self):
        return UISettings()

    def getIngestJobSettingsPanel(self, settings):
        self.settings = settings
        return UISettingsPanel(self.settings)

# Data Source-level ingest module.  One gets created per data source.
class FaceModule(DataSourceIngestModule):

    _logger = Logger.getLogger(FaceModuleFactory.moduleName)

    def log(self, level, msg):
        self._logger.logp(level, self.__class__.__name__, inspect.stack()[1][3], msg)

    def __init__(self, settings):
        self.context = None
        self.localSettings = settings
        self.extensions = []
        self.deleteAfter = False
        # if any error code happens and is not listed, try to run in comandline
        self.errorListDetection = { 
        1: ' .exe Parameters error ', 
        2: ' given path doesnt exist ',
        3: ' no images given ',
        4: ' error loading detector ',
        5: ' error parsing resolution ',
        6: ' cuda out of memory '
        }
        self.errorListRecognition = { 
        1: ' .exe Parameters error ', 
        2: ' error opening facetofind folder ',
        3: ' no images given/found ',
        4: ' error loading detectors ',
        5: ' error loading images ',
        6: ' images to find had more than 1 face or no face detected ',
        7: ' cuda error or network parsing, check recognition detector '
        }
    # Where any setup and configuration is done
    # 'context' is an instance of org.sleuthkit.autopsy.ingest.IngestJobContext.
    # See: http://sleuthkit.org/autopsy/docs/api-docs/4.4/classorg_1_1sleuthkit_1_1autopsy_1_1ingest_1_1_ingest_job_context.html
    def startUp(self, context):
        # Throw an IngestModule.IngestModuleException exception if there was a problem setting up
        # raise IngestModuleException("Oh No!")

        # Supported file format from dlib
        acceptedFiles = ['.jpg', '.jpeg', '.png']
        i = 0
        for ext in acceptedFiles:
            if self.localSettings.getFlag(i):
                self.extensions.append(ext)
            i += 1

        if not self.extensions:
            raise IngestModuleException("Need to select at least one type of file!")

        if self.localSettings.getFlag(3):
            self.deleteAfter = True

        #
        # Checking for default detectors and auxiliary files
        # 
        self.pathToExe = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Facial_Detection.exe")
        if not os.path.exists(self.pathToExe):
            raise IngestModuleException("Executable not in expected place!")

        self.detector =  os.path.join(os.path.dirname(os.path.abspath(__file__)), "mmod_human_face_detector.dat")
        if not os.path.exists(self.detector):
            raise IngestModuleException("Detector not in expected pace!")

        self.pathToExeRec = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Facial_Recognition.exe")
        if not os.path.exists(self.pathToExeRec):
            raise IngestModuleException("Executable not in expected place!")

        self.rec =  os.path.join(os.path.dirname(os.path.abspath(__file__)), "dlib_face_recognition_resnet_model_v1.dat")
        if not os.path.exists(self.rec):
            raise IngestModuleException("Recognitor not in expected pace!")

        self.shape =  os.path.join(os.path.dirname(os.path.abspath(__file__)), "shape_predictor_68_face_landmarks.dat")
        if not os.path.exists(self.shape):
            raise IngestModuleException("Shape predictor not in expected pace!")

        self.context = context

    # Where the analysis is done.
    # The 'dataSource' object being passed in is of type org.sleuthkit.datamodel.Content.
    # See: http://www.sleuthkit.org/sleuthkit/docs/jni-docs/4.4.1/interfaceorg_1_1sleuthkit_1_1datamodel_1_1_content.html
    # 'progressBar' is of type org.sleuthkit.autopsy.ingest.DataSourceIngestModuleProgress
    # See: http://sleuthkit.org/autopsy/docs/api-docs/4.4/classorg_1_1sleuthkit_1_1autopsy_1_1ingest_1_1_data_source_ingest_module_progress.html
    def process(self, dataSource, progressBar):

        # we don't know how much work there is yet
        progressBar.switchToIndeterminate()

        # case insensitive SQL LIKE clause is used to query the case database
        # FileManager API: http://sleuthkit.org/autopsy/docs/api-docs/4.4.1/classorg_1_1sleuthkit_1_1autopsy_1_1casemodule_1_1services_1_1_file_manager.html
        fileManager = Case.getCurrentCase().getServices().getFileManager()

        files = []
        for extension in self.extensions:
            try:
                files.extend(fileManager.findFiles(dataSource, "%" + extension))
            except TskCoreException:
                self.log(Level.INFO, "Error getting files from: '" + extension + "'")
       
        numFiles = len(files)
        if not numFiles:
            self.log(Level.WARNING, "Didn't find any usable files!")
            return IngestModule.ProcessResult.OK

        # Check if the user pressed cancel while we were busy
        if self.context.isJobCancelled():
            return IngestModule.ProcessResult.OK

        self.log(Level.INFO, "Got " + str(numFiles) + " images!")
        
        tempDir = Case.getCurrentCase().getTempDirectory()
        #
        # Copying the files to temp directory
        #
        try:
            os.mkdir(tempDir + "\\" + dataSource.getName())
            i = 0
            for file in files:
                # Checking if we didn't got any currupted files
                if file.getSize() > 0:
                    filename, file_extension = os.path.splitext(file.getName())
                    ContentUtils.writeToFile(file, File(tempDir + "\\" + dataSource.getName() + "\\" + str(i) + file_extension))
                i+=1
        except:
             self.log(Level.INFO, "Directory already exists for this data source skipping file copy")
       
        # Location of data to search
        source = tempDir + "\\" + dataSource.getName()
        # Location where the output of executable will appear
        outFile = source + "\\facesFound.txt"

        if os.path.exists(outFile):
            os.remove(outFile)

        returnCode = 0
        try:
            #
            # Blocking call, we will wait until it finishes which will take a while
            #
            returnCode = subprocess.call([self.pathToExe, source, outFile, self.detector])
        except OSError:
            self.log(Level.SEVERE, "Couldn't run Facial_Detection.exe!")
            return IngestModule.ProcessResult.OK

        if returnCode:
            if returnCode <= len(self.errorListDetection):
                self.log(Level.SEVERE, self.errorListDetection[returnCode])
            else :
                self.log(Level.SEVERE, "unknown error ocurred in Facial_Detection.exe! it returned: " + str(returnCode))
            if self.deleteAfter:
                self.deleteFiles(tempDir + "\\" + dataSource.getName())
            return IngestModule.ProcessResult.ERROR

        self.log(Level.INFO, "Face detection terminated with no problems")  

        # Checking if cancel was pressed before starting another job
        if self.context.isJobCancelled():
            return IngestModule.ProcessResult.OK

        outRec = source + "\\ImagesWithEspecificFace.txt"

        # Use blackboard class to index blackboard artifacts for keyword search
        blackboard = Case.getCurrentCase().getServices().getBlackboard()
        artifactType = BlackboardArtifact.ARTIFACT_TYPE.TSK_INTERESTING_FILE_HIT

        if self.localSettings.getFace():
            self.log(Level.INFO, "Looking for person in: " + self.localSettings.getFace())

            if os.path.exists(outRec):
                os.remove(outRec)
            try:
                #
                # Blocking call, we will wait until it finishes
                #  
                returnCode = subprocess.call([self.pathToExeRec, source, self.localSettings.getFace(), self.shape, self.rec, outRec])
            except OSError:
                self.log(Level.SEVERE, "Couldn't run Facial_Recognition.exe!")
                return IngestModule.ProcessResult.OK

            if returnCode:
                if returnCode <= len(self.errorListRecognition):
                    self.log(Level.SEVERE, self.errorListRecognition[returnCode])
                else :
                    self.log(Level.SEVERE, "unknown error ocurred in Facial_Recognition.exe! it returned: " + str(returnCode))
                if self.deleteAfter:
                    self.deleteFiles(tempDir + "\\" + dataSource.getName())
                return IngestModule.ProcessResult.ERROR

            self.log(Level.INFO, "Face recognition terminated with no problems") 
            

            with open(outRec, "r") as out:

                for line in out:

                    data = line.split('.')
                    pos = int(data[0])

                    interestingFile = files[pos]

                    artifactList = interestingFile.getArtifacts(artifactType)

                    if artifactList:
                        self.log(Level.INFO, "Artifact already exists! ignoring")
                    else:
                        # Make an artifact on the blackboard.  TSK_INTERESTING_FILE_HIT is a generic type of
                        # artfiact.  Refer to the developer docs for other examples.
                        art = interestingFile.newArtifact(artifactType)

                        att = BlackboardAttribute(BlackboardAttribute.ATTRIBUTE_TYPE.TSK_SET_NAME, FaceModuleFactory.moduleName, "Wanted face founded in")
                        art.addAttribute(att)

                        try:
                            # index the artifact for keyword search
                            blackboard.indexArtifact(art)
                        except Blackboard.BlackboardException as e:
                            self.log(Level.SEVERE, "Error indexing artifact " + art.getDisplayName())

        else:
            self.log(Level.INFO, "No Positive folder given, will only do detection") 

        # Parse output file for files with faces and mark them as interesting
        count = 0
        with open(outFile, "r") as out:

            for line in out:
                count += 1

                data = line.split('.')
                pos = int(data[0])

                interestingFile = files[pos]

                artifactList = interestingFile.getArtifacts(artifactType)
                if artifactList:
                    self.log(Level.INFO, "Artifact already exists! ignoring")
                else:
                    # Make an artifact on the blackboard.  TSK_INTERESTING_FILE_HIT is a generic type of
                    # artfiact.  Refer to the developer docs for other examples.
                    art = interestingFile.newArtifact(artifactType)
                    att = BlackboardAttribute(BlackboardAttribute.ATTRIBUTE_TYPE.TSK_SET_NAME, FaceModuleFactory.moduleName, "Image with faces")
                    art.addAttribute(att)

                    try:
                        # index the artifact for keyword search
                        blackboard.indexArtifact(art)
                    except Blackboard.BlackboardException as e:
                        self.log(Level.SEVERE, "Error indexing artifact " + art.getDisplayName())

        if self.deleteAfter:
            self.deleteFiles(tempDir + "\\" + dataSource.getName())

        message = IngestMessage.createMessage(IngestMessage.MessageType.DATA,
            "Face Detector Data Source Ingest Module", "Found %d images with faces" % count)
        IngestServices.getInstance().postMessage(message)

        return IngestModule.ProcessResult.OK

    def deleteFiles(self, path):
        # ignoring the error if the directory is empty
        shutil.rmtree(path, ignore_errors=True)

class UISettings(IngestModuleIngestJobSettings):
    serialVersionUID = 1L

    def __init__(self):
        #             JPG   JPEG  PNG   Delete file after
        self.flags = [True, True, True, False]
        self.faceToFind = ""
                    

    def getVersionNumber(self):
        return serialVersionUID

    def getFlag(self, pos):
        return self.flags[pos]

    def setFlag(self, flag, pos):
        self.flags[pos] = flag

    def getFace(self):
        return self.faceToFind

    def setFace(self, folder):
        self.faceToFind = folder

class UISettingsPanel(IngestModuleIngestJobSettingsPanel):

    def __init__(self, settings):
        self.localSettings = settings
        self.initComponents()
        self.customizeComponents()

    def checkBoxEvent(self, event):
        self.localSettings.setFlag(self.checkboxJPG.isSelected(), 0)
        self.localSettings.setFlag(self.checkboxJPEG.isSelected(), 1)
        self.localSettings.setFlag(self.checkboxPNG.isSelected(), 2)
        self.localSettings.setFlag(self.checkboxDelete.isSelected(), 3)

    def onClick(self, e):
        fileChooser = JFileChooser()
        fileChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY)
        ret = fileChooser.showDialog(self.panel, "Choose folder")
       
        if ret == JFileChooser.APPROVE_OPTION:
            file = fileChooser.getSelectedFile()
            filename = file.getCanonicalPath()
            self.localSettings.setFace(filename)

    def initComponents(self):
        self.setLayout(BoxLayout(self, BoxLayout.Y_AXIS))
        self.setAlignmentX(JComponent.LEFT_ALIGNMENT)
        self.panel = JPanel()
        self.panel.setLayout(BoxLayout(self.panel, BoxLayout.Y_AXIS))
        self.panel.setAlignmentY(JComponent.LEFT_ALIGNMENT)

        self.labelTop = JLabel("Choose file extensions to look for:")
        self.panel.add(self.labelTop)
        self.checkboxJPG = JCheckBox(".jpg", actionPerformed=self.checkBoxEvent)
        self.panel.add(self.checkboxJPG)
        self.checkboxJPEG = JCheckBox(".jpeg", actionPerformed=self.checkBoxEvent)
        self.panel.add(self.checkboxJPEG)
        self.checkboxPNG = JCheckBox(".png", actionPerformed=self.checkBoxEvent)
        self.panel.add(self.checkboxPNG)
        self.labelBlank = JLabel(" ")
        self.panel.add(self.labelBlank)
        self.checkboxDelete = JCheckBox("Delete files after use", actionPerformed=self.checkBoxEvent)
        self.panel.add(self.checkboxDelete)
        self.labelBlank1 = JLabel(" ")
        self.panel.add(self.labelBlank1)
        self.label = JLabel("Provide folder for trainning face (if none will only do detection)")
        self.openb = JButton("Choose", actionPerformed=self.onClick)
        self.panel.add(self.label)
        self.panel.add(self.openb)

        self.add(self.panel)
        
    def customizeComponents(self):      
        self.checkboxJPG.setSelected(self.localSettings.getFlag(0))
        self.checkboxJPEG.setSelected(self.localSettings.getFlag(1))
        self.checkboxPNG.setSelected(self.localSettings.getFlag(2))
        self.checkboxDelete.setSelected(self.localSettings.getFlag(3))

    def getSettings(self):
        return self.localSettings