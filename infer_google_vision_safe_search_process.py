import copy
from ikomia import core, dataprocess
from google.cloud import vision
import os
import io
import cv2
import pandas as pd


# --------------------
# - Class to handle the algorithm parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferGoogleVisionSafeSearchParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        self.google_application_credentials = ''

    def set_values(self, params):
        # Set parameters values from Ikomia Studio or API
        # Parameters values are stored as string and accessible like a python dict
        # Example : self.window_size = int(params["window_size"])
        self.google_application_credentials = str(params["google_application_credentials"])

    def get_values(self):
        # Send parameters values to Ikomia Studio or API
        # Create the specific dict structure (string container)
        params = {}
        params["google_application_credentials"] = str(self.google_application_credentials)
        return params


# --------------------
# - Class which implements the algorithm
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferGoogleVisionSafeSearch(dataprocess.C2dImageTask):

    def __init__(self, name, param):
        dataprocess.C2dImageTask.__init__(self, name)
        # Add input/output of the algorithm here
        self.add_output(dataprocess.DataDictIO())


        # Create parameters object
        if param is None:
            self.set_param_object(InferGoogleVisionSafeSearchParam())
        else:
            self.set_param_object(copy.deepcopy(param))

        self.client = None
        # Names of likelihood from google.cloud.vision.enums
        self.likelihood_name = (
            "UNKNOWN",
            "VERY_UNLIKELY",
            "UNLIKELY",
            "POSSIBLE",
            "LIKELY",
            "VERY_LIKELY",
        )

    def get_progress_steps(self):
        # Function returning the number of progress steps for this algorithm
        # This is handled by the main progress bar of Ikomia Studio
        return 1

    def run(self):
        self.begin_task_run()

        # Get input
        input = self.get_input(0)
        src_image = input.get_image()

        # Set output
        output_dict = self.get_output(1)

        # Get parameters
        param = self.get_param_object()

        if self.client is None:
            if param.google_application_credentials:
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = param.google_application_credentials
            self.client = vision.ImageAnnotatorClient()

        # Convert the NumPy array to a byte stream
        src_image = src_image[..., ::-1] # Convert to bgr
        is_success, image_buffer = cv2.imencode(".jpg", src_image)
        byte_stream = io.BytesIO(image_buffer)

        response = self.client.safe_search_detection(image=byte_stream)

        if response.error.message:
            raise Exception(
                "{}\nFor more info on error messages, check: "
                "https://cloud.google.com/apis/design/errors".format(response.error.message)
            )

        safe = response.safe_search_annotation

        data = {
            "Category": ["adult:", "medical:", "spoofed:", "violence:", "racy:"],
            "Likelihood": [
                self.likelihood_name[safe.adult],
                self.likelihood_name[safe.medical],
                self.likelihood_name[safe.spoof],
                self.likelihood_name[safe.violence],
                self.likelihood_name[safe.racy],
            ],
        }

        df = pd.DataFrame(data)
        print(df.to_string(index=False))

        output_dict.data = {
                            "adult:": self.likelihood_name[safe.adult],
                            "medical:": self.likelihood_name[safe.medical],
                            "spoofed:": self.likelihood_name[safe.spoof],
                            "violence:": self.likelihood_name[safe.violence],
                            "racy:": self.likelihood_name[safe.racy]
                        }

        self.emit_step_progress()

        # Call end_task_run() to finalize process
        self.end_task_run()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferGoogleVisionSafeSearchFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set algorithm information/metadata here
        self.info.name = "infer_google_vision_safe_search"
        self.info.short_description = "Safe Search detects explicit content such as adult content or violent content within an image."
        # relative path -> as displayed in Ikomia Studio algorithm tree
        self.info.icon_path = "images/cloud.png"
        self.info.path = "Plugins/Python/Other"
        self.info.version = "1.0.0"
        self.info.authors = "Google"
        self.info.article = ""
        self.info.journal = ""
        self.info.year = 2023
        self.info.license = "Apache License 2.0"
        # URL of documentation
        self.info.documentation_link = "https://cloud.google.com/vision/docs/detecting-safe-search"
        # Code source repository
        self.info.repository = "https://github.com/Ikomia-hub/infer_google_vision_safe_search"
        self.info.original_repository = "https://github.com/googleapis/python-vision"
        # Keywords used for search
        self.info.keywords = "Safe Search,Google,Cloud,Vision AI"
        self.info.algo_type = core.AlgoType.INFER
        self.info.algo_tasks = "OTHER"


    def create(self, param=None):
        # Create algorithm object
        return InferGoogleVisionSafeSearch(self.info.name, param)
