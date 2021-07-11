import json
from Environment import CustomerClass

class Scenario(object):
    def __init__(self, scenario_json):
        with open("scenarios/" + scenario_json + ".json") as f:
            self.data = json.load(f)
        
        self.prices = self.data['prices']
        self.bids = self.data['bids']
        self.rounds_horizon = self.data['rounds_horizon']
        self.returns_horizon = self.data['returns_horizon']

        if len(self.data['features']) != len(set(self.data['features'])):
            raise Exception("Duplicate features")

        self.features = self.data['features']

        self.customer_classes = []
        for customer_class in self.data['customer_classes']:
            if min(self.prices) > min(customer_class['conversion_function_interpolation_values']['x']) or max(self.prices) < max(customer_class['conversion_function_interpolation_values']['x']):
                raise Exception("Conversion prices out of range")

            if min(self.prices) != min(customer_class['conversion_function_interpolation_values']['x']) or max(self.prices) != max(customer_class['conversion_function_interpolation_values']['x']):
                raise Exception("Missing price bounds for conversion function")

            if min(customer_class['conversion_function_interpolation_values']['y']) < 0 or max(customer_class['conversion_function_interpolation_values']['y']) > 1:
                raise Exception("Invalid conversion probability values")

            if len(self.features) != len(customer_class['feature_values']):
                raise Exception("Feature labels and values not matching")

            if customer_class['returns_function_params']['min'] < 0 or customer_class['returns_function_params']['max'] > 30:
                raise Exception("Invalid range for returns function")

            if customer_class['cost_per_click_function_params']['coefficient'] < 0 or customer_class['cost_per_click_function_params']['coefficient'] > 1:
                raise Exception("Invalid cost per click coefficient")
            
            self.customer_classes.append(CustomerClass(feature_labels=self.features, **customer_class))
        