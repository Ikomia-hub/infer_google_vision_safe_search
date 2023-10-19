from ikomia import dataprocess


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits PyDataProcess.CPluginProcessInterface from Ikomia API
# --------------------
class IkomiaPlugin(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def get_process_factory(self):
        # Instantiate algorithm object
        from infer_google_vision_safe_search.infer_google_vision_safe_search_process import InferGoogleVisionSafeSearchFactory
        return InferGoogleVisionSafeSearchFactory()

    def get_widget_factory(self):
        # Instantiate associated widget object
        from infer_google_vision_safe_search.infer_google_vision_safe_search_widget import InferGoogleVisionSafeSearchWidgetFactory
        return InferGoogleVisionSafeSearchWidgetFactory()
