from itertools import combinations
import networkx as nx
import random,copy
import numpy as np
from scipy.stats import rankdata
import json
import statistics

class Hypergraph:
    def __init__(self, hygraph_path=None, max_size=10, r=3):
        self.max_size = max_size
        if hygraph_path is None:
            self.edges = {}
            self.nodes = {}
            self.r = r
        else:
            with open(hygraph_path, 'r') as json_file:
                data = json.load(json_file)
            self.edges = data['edges']
            self.nodes = data['nodes']
            self.r = data['r']
            print("Build graph from json file: {}".format(self.edges))
    
    def degree(self):
        degree = {}
        for edge in self.edges.keys():
            num = len(edge) - 1
            for node in edge:
                if node in degree.keys():
                    degree[node] += num
                else:
                    degree[node] = num
        return degree
    
    def print_info(self):
        print("************* The hypergraph *************")
        print(f"It has {len(self.nodes)} nodes: {self.nodes}")
        for e,w in self.edges.items():
            print(f"edge : {e} with weight {w}.")
        print("************* The hypergraph *************")
         
    def save(self, path):
        graph = {"edges": self.edges, "nodes": self.nodes, "r": self.r}
        with open(path, 'w') as json_file:
            json.dump(graph, json_file)
        print("saved to {}".format(path))

    def is_edge_in(self, edge):
        for e in self.edges.keys():
            if sorted(e) == sorted(edge):
                return True, e
        return False,None

    def add_edge(self, edge, weight=1):
        flag, key = self.is_edge_in(edge)
        if flag:
            self.edges[key] = weight
        else:
            self.edges[edge] = weight
            for node in list(edge):
                if node in self.nodes.keys():
                    self.nodes[node] += 1
                else:
                    self.nodes[node] = 1
                    
    def del_edge(self, edge):
        flag, key = self.is_edge_in(edge)
        if flag:
            self.edges.pop(edge)
            del_nodes = list(edge)
            for node in del_nodes:
                if node in self.nodes.keys():
                    if self.nodes[node] == 1:
                        self.nodes.pop(node)
                    else:
                        self.nodes[node] -= 1
        else:
            print("The edge is not in the hypergraph")

    def get_edge_weight(self, nodes):
        edge = ''.join(nodes)
        for e in self.edges.keys():
            if sorted(e) == sorted(edge):
                return e, self.edges[e]
        return None, None

    def copy_train_edge(self, min_node, team_1, team_2):
        old_edge, old_weight = self.get_edge_weight(min_node, team_1, team_2)
        if old_edge is None:
            1/0
        # hard coding here
        new_edge, new_weight = self.get_edge_weight(self.max_size-1, team_1, team_2)
        if old_edge is None:
            1/0
        print("copy edge from {} - value: {} to {} - value : {}".format(new_edge, new_weight, old_edge, old_weight))
        self.edges[old_edge] = new_weight

    def del_node(self, node):
        #bug here
        edges = copy.deepcopy(list(self.edges.keys()))
        for edge in edges:
            if node in edge:
                self.del_edge(edge)

    def normalize(self, prob_array):
        min_value = np.min(prob_array)
        if min_value < 0:
            prob_array = prob_array - min_value + 1e-6

        # If the sum is zero (e.g., if all values were zero or negative),
        # we handle this by setting a uniform distribution
        total = np.sum(prob_array)
        if total == 0:
            # Set to a uniform distribution if all probabilities were initially zero
            prob_array = np.ones_like(prob_array) / len(prob_array)
        else:
            # Normalize the probabilities to sum to 1
            prob_array /= total

        return prob_array

    def sample_teammates(self, self_id=None, num=2, ablation=False):

        value = self.myerson_value()
        if self_id is not None:
            value.pop(str(self_id))
        sorted_name = sorted(value.keys())
        if ablation:
            sampling_prob_np = np.array([1/len(sorted_name) for name in sorted_name])
        else:
            sampling_prob_np = np.array([value[name] for name in sorted_name])
            sampling_prob_np = self.normalize(sampling_prob_np)
            # if random.random() < p:
            #     # half self-play half train
            #     if random.random() < 0.5:
            #         print("Sample teammate: Self play mode....")
            #         return None
            #     else:
            #         print("Sample teammate: expert mode....")
            # else:
            #     print("Sample teammate: inverse mode....")
            # sampling_prob_np = 1 / sampling_prob_np

            # sampling_rank_np = rankdata(sampling_prob_np, method='dense')
            # sampling_prob_np = sampling_rank_np / sampling_rank_np.sum()
            # sampling_prob_np = sampling_prob_np / sampling_prob_np.sum()
        # print(f"the sampling_prob_np is {sampling_prob_np}")
        team_idx = np.random.choice(list(range(len(sorted_name))), num, replace=False, p=sampling_prob_np)

        if self_id is not None:
            idxs = []
            for idx in team_idx:
                if idx < self_id:
                    idxs.append(idx)
                else:
                    idxs.append(idx+1)
            print(f"the sampled team_idx is {idxs} and self index is {self_id}")
            return idxs
        else:
            # print(f"the sampled team_idx is {team_idx}")
            return team_idx
        
    def sample_teammates_W(self, self_id=None, num=2):
        total_value = {node: 0 for node in self.nodes.keys()}
        node_count = {node: 0 for node in self.nodes.keys()}
        for edge in self.edges.keys():
            for node in edge:
                total_value[node] += self.edges[edge]
                node_count[node] += 1
        value = total_value
        if self_id is not None:
            value.pop(str(self_id))
        sorted_name = sorted(value.keys())
        sampling_prob_np = np.array([value[name] for name in sorted_name])
        # sampling_prob_np = self.normalize(sampling_prob_np)
        # sampling_prob_np = 1 / sampling_prob_np
        sampling_prob_np = self.normalize(sampling_prob_np)
        # print(f"the sampling_prob_np is {sampling_prob_np}")
        team_idx = np.random.choice(list(range(len(sorted_name))), num, replace=False, p=sampling_prob_np)

        if self_id is not None:
            idxs = []
            for idx in team_idx:
                if idx < self_id:
                    idxs.append(idx)
                else:
                    idxs.append(idx+1)
            print(f"the sampled team_idx is {idxs} and self index is {self_id}")
            return idxs
        else:
            print(f"the sampled team_idx is {team_idx} and self index is {self_id}")
            return team_idx

class Preference_Graph():
    def __init__(self,hypergraph):
        self.edges = {}
        self.max_size = hypergraph.max_size
        self.hypergraph = hypergraph
        self.nodes = self.hypergraph.nodes.keys()
        for node in self.hypergraph.nodes.keys():
            max_weight = -np.inf
            max_edge = None
            for edge, weight in self.hypergraph.edges.items():
                if node in list(edge):
                    if weight > max_weight:
                        max_edge = edge
                        max_weight = weight
            if node in self.edges.keys():
                0/1
            else:
                self.edges[node] = max_edge
                
    def normalize(self, prob_array):
        min_value = np.min(prob_array)
        if min_value < 0:
            prob_array = prob_array - min_value + 1e-6

        # If the sum is zero (e.g., if all values were zero or negative),
        # we handle this by setting a uniform distribution
        total = np.sum(prob_array)
        if total == 0:
            # Set to a uniform distribution if all probabilities were initially zero
            prob_array = np.ones_like(prob_array) / len(prob_array)
        else:
            # Normalize the probabilities to sum to 1
            prob_array /= total

        return prob_array
    
    def in_degree(self):
        in_degree = {}
        for node, edge in self.edges.items():
            others = set(edge) - set(node)
            for n in others:
                if n in in_degree.keys():
                    in_degree[n] += 1
                else:
                    in_degree[n] = 1
        for node in self.nodes:
            if node not in in_degree.keys():
                in_degree[node] = 1/3
        return in_degree

    def eta(self):
        G_degree = self.hypergraph.degree()
        eta_value = {}
        for node, in_degree in self.in_degree().items():
            eta_value[node] = in_degree / G_degree[node]
        return eta_value
    
    def min_eta(self):
        value = self.eta()
        sorted_name = sorted(value.keys())
        eta_values = np.array([value[name] for name in sorted_name])
        # 最大值及其索引
        min_value = np.min(eta_values)
        min_index = np.random.choice(np.where(eta_values == min_value)[0])
        print("eats value is {}, del index is {}".format(eta_values, min_index))
        return min_index
    
    def eval_teammates(self):
        value = self.eta()
        sorted_name = sorted(value.keys())
        eta_values = np.array([value[name] for name in sorted_name])
        # 最大值及其索引
        max_value = np.max(eta_values)
        max_index = np.where(eta_values == max_value)[0][-1]  # 返回最大值的索引

        # 中位数及其索引
        median_value = statistics.median_high(eta_values)
        median_index = np.where(eta_values == median_value)[0][-1]  # 找到中位数的索引
        return median_index, max_index
    
    def sample(self, num, debug=False):
        value = self.eta()
        sorted_name = sorted(value.keys())
        sampling_prob_np = np.array([value[name] for name in sorted_name])
        # sampling_prob_np = self.normalize(sampling_prob_np)
        if debug:
            print(sampling_prob_np)
        sampling_prob_np = 1 / sampling_prob_np
        if debug:
            print(sampling_prob_np)
        sampling_prob_np = self.normalize(sampling_prob_np)
        if debug:
            print(sampling_prob_np)
        team_idx = np.random.choice(sorted_name, num, replace=False, p=sampling_prob_np)
        return team_idx
    
    def is_ok(self, idx, k=5):
        th = (self.max_size - k)/2
        def value_rank(data, key, start_idx):
            # Check if the key exists in the dictionary
            if key not in data.keys():
                print(f"{key} is not in {data.keys()}")
                1/0
                return None  # or raise an Exception, based on your preference

            # Filter the dictionary to include only keys greater than start_idx
            filtered_data = {k: v for k, v in data.items() if int(k) > start_idx}

            # Sort the filtered dictionary by value in descending order
            sorted_data = sorted(filtered_data.items(), key=lambda item: item[1], reverse=True)

            # Find the rank of the specified key
            rank = None
            for idx, (k, v) in enumerate(sorted_data):
                if k == key:
                    rank = idx + 1
                    break
            target_value = data[key]
            print(f"eta value is {sorted_data}, the trainer value is {target_value}, the rank is {rank}")

            return rank

        eta_value = self.eta()
        # Get the value corresponding to the key
        rank_num = value_rank(eta_value, str(idx), k)
        if rank_num > th:
            print(f"The check is False, the rank should be within {th}, but it is {rank_num}")
            return False, rank_num
        else:
            return True, rank_num


if __name__ == '__main__':
    # Create an empty hypergraph
    H = Hypergraph()

    # Adding hyperedges from a dictionary where keys are hyperedges and values are weights
    # hypergraph_data = {'201': 61.55072594044716, '301': 53.76279981936254, '302': 36.551187488017476, '312': 70.98842472362145,
    #  '401': 76.86687522388047, '402': 76.68495290166268, '403': 12.739889531906144, '412': 64.72920952884903,
    #  '413': -14.082007707020134, '423': -8.291293921279495, '501': 41.14160076333179, '502': 72.4384613544829,
    #  '503': 37.384042361503376, '504': 58.845805384544235, '512': 62.695171713530556, '513': 11.264499501276063,
    #  '514': 21.460797848162997, '523': -14.18501754282099, '524': 36.46626859969899, '534': 64.06809009466603,
    #  '601': 68.31303121858095, '602': 23.6784977058672, '603': 78.341095835275, '604': 53.71114529920207,
    #  '605': 88.68645841492858, '612': 30.58377870313575, '613': 86.37177972485534, '614': 16.651822723916407,
    #  '615': 80.67870999833474, '623': 74.76014163003589, '624': 53.1567933231034, '625': 93.1582526731184,
    #  '634': 54.75713723377845, '635': 53.7908249107293, '645': 66.79039990239464, '701': 52.18741359386605,
    #  '702': 41.32852779014409, '703': 82.45819917295893, '704': 39.23165997456849, '705': 43.32592708436624,
    #  '706': 43.27022118857546, '712': 45.1168899395793, '713': 45.84212047839517, '714': 41.17163850635396,
    #  '715': 33.188001136577284, '716': 55.62585327768181, '723': 48.843777460106196, '724': 37.87568986817749,
    #  '725': 63.446918990936446, '726': 10.508900313878426, '734': 82.77459489044983, '735': 43.733476039003776,
    #  '736': 68.01595506675575, '745': 42.505029091713546, '746': 70.45336097227275, '756': 42.480107533968614,
    #  '801': 66.75140427602854, '802': 86.51947189361793, '803': 74.97787525644637, '804': 67.5835392170844,
    #  '805': 48.81626971085201, '806': 62.6935674531555, '807': 32.742379854376125, '812': 36.681068248631405,
    #  '813': 32.1202290487428, '814': 47.766185547132736, '815': 71.93529744017242, '816': 11.934790193532555,
    #  '817': 72.84483582199442, '823': 21.54462832614063, '824': 45.87872568364527, '825': 55.48204528755738,
    #  '826': 55.56674768361602, '827': 55.647540615338094, '834': 65.97410072865947, '835': 31.66961261010345,
    #  '836': 58.678289857211865, '837': 76.74765548349131, '845': 50.852628223739934, '846': 86.87477101766501,
    #  '847': 73.55734323822489, '856': 27.029357972031967, '857': 49.59835871024501, '867': 59.13719412960594,
    #  '901': 30.703027003645815, '902': 35.804916780554606, '903': 11.052180996324681, '904': 6.189641791983793,
    #  '905': 13.974143357551842, '906': 17.264008369254768, '907': 41.94225880305307, '908': 61.744672713380226,
    #  '912': 58.76387605084788, '913': -10.965667647756556, '914': 8.57227897621908, '915': 11.861293020641494,
    #  '916': 37.45057459518696, '917': 55.0679652909024, '918': 41.89864372897543, '923': -2.3132073846787633,
    #  '924': -6.931899670227685, '925': 7.532892302260409, '926': 38.80133648009319, '927': 69.04954177437493,
    #  '928': 65.59936209016891, '934': 10.745051230425004, '935': 29.1050959683294, '936': 39.504243468874705,
    #  '937': 62.282893617189494, '938': 58.447295009007654, '945': 43.28653603549913, '946': 38.722278762408266,
    #  '947': 58.231096998238996, '948': 58.010573780478595, '956': 36.769832343337725, '957': 30.934510774763577,
    #  '958': 68.39255271129765, '967': 37.43273423946314, '968': 44.67205219976269, '978': 41.42442004484521}

    hypergraph_data = {
        '124': -0.01,
        '103': 0.17,
        '123': 1.49,
        '102': 4.62,
        '134': -0.03,
        '104': 21.04
    }
    #{'201': 61.55072594044716, '301': 53.76279981936254, '302': 36.551187488017476, '312': 70.98842472362145, '401': 76.86687522388047}#, '402': 76.68495290166268, '403': 12.739889531906144, '412': 64.72920952884903, '413': -14.082007707020134, '423': -8.291293921279495, '501': 41.14160076333179, '502': 72.4384613544829, '503': 37.384042361503376, '504': 58.845805384544235, '512': 62.695171713530556, '513': 11.264499501276063, '514': 21.460797848162997, '523': -14.18501754282099, '524': 36.46626859969899, '534': 64.06809009466603, '601': 68.31303121858095, '602': 23.6784977058672, '603': 78.341095835275, '604': 53.71114529920207, '605': 88.68645841492858, '612': 30.58377870313575, '613': 86.37177972485534, '614': 16.651822723916407, '615': 80.67870999833474, '623': 74.76014163003589, '624': 53.1567933231034, '625': 93.1582526731184, '634': 54.75713723377845, '635': 53.7908249107293, '645': 66.79039990239464, '701': 52.18741359386605, '702': 41.32852779014409, '703': 82.45819917295893, '704': 39.23165997456849, '705': 43.32592708436624, '706': 43.27022118857546, '712': 45.1168899395793, '713': 45.84212047839517, '714': 41.17163850635396, '715': 33.188001136577284, '716': 55.62585327768181, '723': 48.843777460106196, '724': 37.87568986817749, '725': 63.446918990936446, '726': 10.508900313878426, '734': 82.77459489044983, '735': 43.733476039003776, '736': 68.01595506675575, '745': 42.505029091713546, '746': 70.45336097227275, '756': 42.480107533968614, '801': 66.75140427602854, '802': 86.51947189361793, '803': 74.97787525644637, '804': 67.5835392170844, '805': 48.81626971085201, '806': 62.6935674531555, '807': 32.742379854376125, '812': 36.681068248631405, '813': 32.1202290487428, '814': 47.766185547132736, '815': 71.93529744017242, '816': 11.934790193532555, '817': 72.84483582199442, '823': 21.54462832614063, '824': 45.87872568364527, '825': 55.48204528755738, '826': 55.56674768361602, '827': 55.647540615338094, '834': 65.97410072865947, '835': 31.66961261010345, '836': 58.678289857211865, '837': 76.74765548349131, '845': 50.852628223739934, '846': 86.87477101766501, '847': 73.55734323822489, '856': 27.029357972031967, '857': 49.59835871024501, '867': 59.13719412960594, '901': 29.342209105537997, '902': 51.72554419616979, '903': 1.6843510449999148, '904': 3.599409891557002, '905': 15.019780387109918, '906': 46.937619399415695, '907': 42.81320123950901, '908': 54.72405212230878, '912': 76.77754581558422, '913': -1.5449714587384658, '914': 14.609648384974408, '915': -2.096931153588576, '916': 38.63942884487655, '917': 31.791698441065424, '918': 59.84670946321305, '923': -5.363806221541171, '924': -3.732998528846001, '925': 25.466490007339182, '926': 21.778333351538848, '927': 36.72411870333618, '928': 41.72695627716115, '934': 5.285704131766201, '935': 42.87718133234325, '936': 58.55740855547429, '937': 59.52512031544407, '938': 32.267770574535476, '945': 39.262902809282394, '946': 47.757405776449204, '947': 18.696637178403993, '948': 34.72482319060445, '956': 36.283851195507545, '957': 73.49716791967002, '958': 37.34240394651363, '967': 39.37297558912514, '968': 31.26650268063181, '978': 36.719072194324696}
    for edge, weight in hypergraph_data.items():
        # Add hyperedge with weight as an attribute
        H.add_edge(edge, weight=weight)
    # H.sample_teammates_W()
    # H.sample_teammates()
    # Display the hypergraph
    # for edge in H.edges.keys():
    #     print(f"Hyperedge {edge}: Weight = {H.edges[edge]}, Connects nodes {list(edge)}")
    # Calculate Myerson value
    # print("Myerson Values:", H.myerson_value())
    #
    pg = Preference_Graph(H)
    print(pg.edges.items())
    value = pg.eta()
    print(value)
    a = pg.min_eta()
    print(a)
