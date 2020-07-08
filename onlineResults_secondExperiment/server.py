import os
import math
import datetime
import pickle
import numpy as np
import pandas as pd
import sqlite3


from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import euclidean_distances, pairwise_distances
from collections import defaultdict, OrderedDict
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse
from cgi import parse_header, parse_multipart

class Recommender:
    # run initialization of all recommending models utilized in online evaluation
    # always keep only the resulting model, dictionary and rev_dict
    def save_obj(self, obj, name):
        with open('obj/' + name + '.pkl', 'wb') as f:
            pickle.dump(obj, f)

    def load_obj(self, name):
        with open('obj/' + name + '.pkl', 'rb') as f:
            return pickle.load(f)

    def mmr_objects_similarity(self, i, j, rev_dict):
        try:
            idi = self.cbDict[rev_dict[i]]
            idj = self.cbDict[rev_dict[j]]
            return self.dfCBSim[idi, idj]
        except:
            return 0

    def mmr_sorted(self, docs, lambda_, results, rev_dict, length):
        """Sort a list of docs by Maximal marginal relevance

        Performs maximal marginal relevance sorting on a set of
        documents as described by Carbonell and Goldstein (1998)
        in their paper "The Use of MMR, Diversity-Based Reranking
        for Reordering Documents and Producing Summaries"

        :param docs: a set of documents to be ranked
                      by maximal marginal relevance
        :param q: query to which the documents are results
        :param lambda_: lambda parameter, a float between 0 and 1
        :return: a (document, mmr score) ordered dictionary of the docs
                given in the first argument, ordered my MMR
        """
        # print("enter to MMR")
        selected = OrderedDict()
        docs = set(docs)
        while (len(selected) < len(docs)) and (len(selected) < length):
            remaining = docs - set(selected)
            mmr_score = lambda x: lambda_ * results[x] - (1 - lambda_) * max(
                [self.mmr_objects_similarity(x, y, rev_dict) for y in set(selected) - {x}] or [
                    0])  # TODO: self.mmr_objects_similarity
            next_selected = self.argmax(remaining, mmr_score)
            selected[next_selected] = len(selected)
            # print(len(selected))
        return selected

    def argmax(self, keys, f):
        return max(keys, key=f)

    def init_recommending_models(self, algDesc):
        for alg in algDesc:
            if len(self.alg_models[(alg["name"], alg["params"])]) == 0:
                # load model to dictionary
                model = self.load_obj(alg["name"] + "_" + alg["params"] + "_model")
                if alg["name"] == "word2vec":
                    dictionary = self.load_obj(alg["name"] + "_" + alg["params"] + "_dict")
                    rev_dict = self.load_obj(alg["name"] + "_" + alg["params"] + "_revdict")
                else:
                    dictionary = self.load_obj(alg["name"] + "_dict")
                    rev_dict = self.load_obj(alg["name"] + "_revdict")
                self.alg_models[(alg["name"], alg["params"])] = [model, dictionary, rev_dict]
                # TODO: create models from full input data

    def __init__(self):
        self.alg_models = defaultdict(list)
        self.algorithm_descriptions = [
            {"name": "best_old", "params": "ctr", "aggregation": "", "novelty": False, "diversity": False},
            {"name":"vsm", "params": "sameAllowed", "aggregation": "mean", "novelty": False, "diversity": True},
            {"name":"word2vec", "params": "32_5", "aggregation": "temporal", "novelty": False, "diversity": True},
            {"name": "best_current", "params": "ctr", "aggregation": "", "novelty": False, "diversity": False},
            {"name": "best_old", "params": "vrr", "aggregation": "", "novelty": False, "diversity": False},
            {"name": "best_current", "params": "vrr", "aggregation": "", "novelty": False, "diversity": False}
            
        ]
        # phase 1 settings
        #    {"name":"vsm", "params": "sameAllowed", "aggregation": "mean", "novelty": False, "diversity": True},
        #    {"name":"word2vec", "params": "32_3", "aggregation": "window10", "novelty": False, "diversity": True},
        #    {"name": "dhondt", "params": "", "aggregation": "", "novelty": False, "diversity": False},
        #    {"name": "bandit", "params": "", "aggregation": "", "novelty": False, "diversity": False},
        
        self.base_algorithms = [
            {"name":"doc2vec", "params": "128_1", "aggregation": "last", "novelty": False, "diversity": True},
            {"name":"doc2vec", "params": "128_1", "aggregation": "temporal", "novelty": True, "diversity": False},
            {"name": "doc2vec", "params": "32_5", "aggregation": "mean", "novelty": False, "diversity": False},
            {"name":"doc2vec", "params": "32_5", "aggregation": "mean", "novelty": True, "diversity": False},
            {"name":"doc2vec", "params": "128_5", "aggregation": "max", "novelty": False, "diversity": True},
            {"name":"vsm", "params": "noSameObjects", "aggregation": "temporal", "novelty": False, "diversity": True},
            {"name": "vsm", "params": "sameAllowed", "aggregation": "mean", "novelty": False, "diversity": True},
            {"name":"vsm", "params": "sameAllowed", "aggregation": "window10", "novelty": False, "diversity": False},
            {"name":"word2vec", "params": "64_5", "aggregation": "mean", "novelty": True, "diversity": False},
            {"name": "word2vec", "params": "32_5", "aggregation": "temporal", "novelty": False, "diversity": True},
            {"name":"word2vec", "params": "128_3", "aggregation": "last", "novelty": False, "diversity": False},
            {"name": "word2vec", "params": "32_3", "aggregation": "window10", "novelty": False, "diversity": False}
        ]

        print("Volume of recommenders: {}".format(len(self.algorithm_descriptions)))

        self.init_recommending_models(self.base_algorithms)

        dfValidDates = pd.read_csv("data/serialValidDates.csv", sep=";", header=0, index_col=0)
        dfValidDates.novelty_date = pd.to_datetime(dfValidDates.novelty_date)
        now = datetime.datetime.now()
        novelty_score = 1 / np.log((now - dfValidDates.novelty_date).dt.days + 2.72)
        # print(novelty_score)
        dfValidDates["noveltyScore"] = novelty_score
        self.dfValidDates = dfValidDates
        dct = defaultdict(int)
        self.noveltyDict = dfValidDates.noveltyScore.to_dict(into=dct)

        dfCBFeatures = pd.read_csv("data/serialCBFeatures.txt", sep=",", header=0, index_col=0)
        self.dfCBSim = 1 - pairwise_distances(dfCBFeatures, metric="cosine")

        cbNames = dfCBFeatures.index.values
        cbVals = range(len(cbNames))
        self.cbDict = dict(zip(cbNames, cbVals))


    def get_params_for_agg_method(self, alg_name):
        import sqlite3
        if alg_name == "bandit":
            conn = sqlite3.connect('hyperparamsDb.sqlite')
            df = pd.read_sql_query("SELECT * FROM banditParams;", conn)
            conn.close()
            return df
        elif alg_name == "dhondt":
            conn = sqlite3.connect('hyperparamsDb.sqlite')
            df = pd.read_sql_query("SELECT * FROM dhondtParams;", conn)
            conn.close()
            return df


    def dhondt_item_params_to_string(self,params):
        res = []
        for (key,val) in params.items():
            res.append(str(key)+":"+str(val))
        return "::".join(res)


    def aggregate_results(self, alg_name, recs, agg_method_params):
        agg_method_params.set_index("method", inplace=True)

        if alg_name == "bandit":
            results = aggr_bandit_ts.aggrBanditTSRun(recs, agg_method_params)
            print(results[0:5])
            results = [(i[0], "bandit::"+i[1]) for i in results]
            return results

        elif alg_name == "dhondt":
            results = aggr_elections.aggrElectionsRunWithResponsibility(recs, agg_method_params)
            print(results[0:5])
            results = [(i[0], "dhondt::" + self.dhondt_item_params_to_string(i[1])) for i in results]
            return results



    def recommend(self, algorithmVariant, userTrainData, userTrainLogDates, allowedOIDs):

        alg = self.algorithm_descriptions[algorithmVariant]
        alg_name = alg["name"]

        
        if alg_name in ["dhondt","bandit"]:
            recs = {}
            
            for i in range(len(self.base_algorithms)):
                alg = self.base_algorithms[i]
                model = self.alg_models[(alg["name"], alg["params"])][0]
                dictionary = self.alg_models[(alg["name"], alg["params"])][1]
                rev_dict = self.alg_models[(alg["name"], alg["params"])][2]
                base_res = self.recommend_base(model, dictionary, rev_dict, userTrainData, userTrainLogDates, alg["name"], alg["aggregation"], alg["diversity"], alg["novelty"], allowedOIDs)
                base_ids = base_res["objects"]
                base_rating = base_res["ratings"]

                recs[alg["name"]+"_"+alg["params"]+"_"+alg["aggregation"]] = pd.Series(base_rating,base_ids,name=str(i) + "_" + alg["name"])

            agg_method_params = self.get_params_for_agg_method(alg_name)

            return self.aggregate_results(alg_name, recs, agg_method_params)
            
        elif alg_name in ["best_current","best_old"]:  
            #static recommendations by the best algorithms w.r.t. off-line to on-line mapping  (either based on current or past results)
            return self.recommend_best(userTrainData, userTrainLogDates, alg["name"], alg["params"], alg["aggregation"], alg["diversity"], alg["novelty"], allowedOIDs)

        elif alg_name in ["word2vec", "doc2vec", "vsm"]:
            #this is a base recommender - just recommend the items

            model = self.alg_models[(alg["name"], alg["params"])][0]
            dictionary = self.alg_models[(alg["name"], alg["params"])][1]
            rev_dict = self.alg_models[(alg["name"], alg["params"])][2]

            base_res =  self.recommend_base(model, dictionary, rev_dict, userTrainData, userTrainLogDates, alg["name"], alg["aggregation"], alg["diversity"], alg["novelty"], allowedOIDs)
            base_ids =  base_res["objects"]
            res = [(i,"base_method_"+str(algorithmVariant)) for i in base_ids][0:20]
            return res
            
            

    def recommend_best(self, userTrainData, userTrainLogDates, alg, params, rec, diversity, novelty, allowedOIDs):               

        best_current_alg = {}
        best_current_alg["ctr"] = {
            1: {"name":"word2vec", "params": "128_3", "aggregation": "mean", "novelty": False, "diversity": False},
            3: {"name":"word2vec", "params": "128_3", "aggregation": "mean", "novelty": False, "diversity": False},
            6: {"name": "doc2vec", "params": "32_5", "aggregation": "mean", "novelty": True, "diversity": False}
        }
        
        best_current_alg["vrr"] = {
            1: {"name":"word2vec", "params": "32_3", "aggregation": "temporal10", "novelty": False, "diversity": True},
            3: {"name":"doc2vec", "params": "32_3", "aggregation": "temporal3", "novelty": False, "diversity": True},
            6: {"name": "doc2vec", "params": "128_5", "aggregation": "max", "novelty": False, "diversity": False}
        }
        
        best_old_alg = {}      
        best_old_alg["ctr"] = {
            1: {"name":"word2vec", "params": "64_5", "aggregation": "mean", "novelty": True, "diversity": False},
            3: {"name":"word2vec", "params": "32_5", "aggregation": "mean", "novelty": False, "diversity": False},
            6: {"name": "word2vec", "params": "32_5", "aggregation": "mean", "novelty": False, "diversity": False}
        }
        best_old_alg["vrr"] = {
            1: {"name":"word2vec", "params": "32_5", "aggregation": "temporal10", "novelty": False, "diversity": False},
            3: {"name":"doc2vec", "params": "32_5", "aggregation": "mean", "novelty": False, "diversity": False},
            6: {"name": "word2vec", "params": "128_1", "aggregation": "max", "novelty": False, "diversity": False}
        }
        
        variant = 1
        if (len(userTrainData) >= 6):
            variant = 6
        elif (len(userTrainData) >= 3):
            variant = 3
        
        if  alg == "best_current":
            baseAlg = best_current_alg[params][variant]
        else:
            baseAlg = best_old_alg[params][variant]
            
        model = self.alg_models[(baseAlg["name"], baseAlg["params"])][0]
        dictionary = self.alg_models[(baseAlg["name"], baseAlg["params"])][1]
        rev_dict = self.alg_models[(baseAlg["name"], baseAlg["params"])][2]
        
        base_res =  self.recommend_base(model, dictionary, rev_dict, userTrainData, userTrainLogDates, baseAlg["name"], baseAlg["aggregation"], baseAlg["diversity"], baseAlg["novelty"], allowedOIDs)
        base_ids =  base_res["objects"]
        res = [(i, alg+"_"+params) for i in base_ids][0:20] 
        return res   


        
    def recommend_base(self, model, dictionary, rev_dict, userTrainData, userTrainLogDates, alg, rec, diversity, novelty, allowedOIDs):
        #TODO: convert results to object, rating pair rather than plain objects
        #print( (userTrainData, userTrainLogDates, alg, rec, diversity, novelty)  )
        
        # remove objects that are no longer valid TODO transform to keep only allowed OIDs
        if len(allowedOIDs) > 0:
            resultsValidity = [i for i in range(len(rev_dict)) if (rev_dict[i] in self.dfValidDates.index) and (
                self.dfValidDates.available_date[rev_dict[i]] > "2020-01-15") and (rev_dict[i] in allowedOIDs)]
        else:
            resultsValidity = [i for i in range(len(rev_dict)) if (rev_dict[i] in self.dfValidDates.index) and (
                self.dfValidDates.available_date[rev_dict[i]] > "2020-01-15")]

        #print(len(resultsValidity))
        try:
            # remove no longer known IDs
            trainModelIDs = list(map(dictionary.get, userTrainData))
            if (rec == "temporal") | (rec == "temporal3") | (rec == "temporal5") | (rec == "temporal10"):
                tw = [userTrainLogDates[i] for i in range(len(userTrainLogDates)) if trainModelIDs[i] is not None]

            trainModelIDs = list(filter(None.__ne__, trainModelIDs))
            userTrainData = list(map(rev_dict.get, trainModelIDs))

        except:
            print("Error")
            userTrainData = []
        #print(len(userTrainData))
        if (len(userTrainData) > 0):
            if (rec == "mean") | (rec == "max"):
                weights = [1.0] * len(userTrainData)
            elif rec == "last":
                # userTrainData = userTrainData[-1:]
                trainModelIDs = trainModelIDs[-1:]
                weights = [1.0]
            elif rec == "window3":
                userTrainData = userTrainData[-3:]
                trainModelIDs = trainModelIDs[-3:]
                weights = [1 / len(userTrainData) * i for i in range(1, (len(userTrainData) + 1))]
            elif rec == "window5":
                userTrainData = userTrainData[-5:]
                trainModelIDs = trainModelIDs[-5:]
                weights = [1 / len(userTrainData) * i for i in range(1, (len(userTrainData) + 1))]
            elif rec == "window10":
                userTrainData = userTrainData[-10:]
                trainModelIDs = trainModelIDs[-10:]
                weights = [1 / len(userTrainData) * i for i in range(1, (len(userTrainData) + 1))]

            elif rec == "temporal3":
                # userTrainData = userTrainData[-3:]
                trainModelIDs = trainModelIDs[-3:]
                weights = [float(i) for i in tw[-3:]]

            elif rec == "temporal5":
                # userTrainData = userTrainData[-5:]
                trainModelIDs = trainModelIDs[-5:]
                weights = [float(i) for i in tw[-5:]]

            elif rec == "temporal10":
                # userTrainData = userTrainData[-10:]
                trainModelIDs = trainModelIDs[-10:]
                weights = [float(i) for i in tw[-10:]]

            elif rec == "temporal":
                weights = [float(i) for i in tw]

            #print(trainModelIDs)
            #print(type(trainModelIDs[0]))
            embeds = model[trainModelIDs]
            if alg == "vsm":  # attributeCosineSim
                results = embeds
            else:
                results = 1 - pairwise_distances(embeds, model, metric="cosine")

            weights = np.asarray(weights).reshape((-1, 1))
            if rec == "max":
                results = np.max(results, axis=0)
            else:
                results = results * weights
                results = np.mean(results, axis=0)

            #print(results[0:20])
            #print(results.shape)

            noveltyList = np.asarray(list(map(self.noveltyDict.get, [rev_dict[i] for i in range(len(rev_dict))])))
            noveltyList = noveltyList[resultsValidity]

            rdKeys = range(len(results))
            rdVals = [rev_dict[i] for i in resultsValidity]
            rev_dict_updated = dict(zip(rdKeys, rdVals))

            results = results[resultsValidity]

            #print(results[0:20])
            #print(results.shape)
            
            if novelty == True:
                results = (0.8 * results) + (0.2 * noveltyList)
            if diversity == True:
                resultList = self.mmr_sorted(range(len(results)), 0.8, results, rev_dict_updated, 10)
                #convert ordered dict to list of indexes
                resultList = [i for i in resultList.keys()]
            else:
                resultList = (-results).argsort()[0:100]
                
            #print(resultList)    
            #print(results[resultList])
            #print([rev_dict_updated[i] for i in resultList])  
            #print([(rev_dict_updated[i],(-i)) for i in resultList])  
            
            #normalize scores into the unit vector (for aggregation purposes)
            finalScores = results[resultList]
            finalScores = normalize(np.expand_dims(finalScores, axis=0))[0,:]
            
            return {"objects":[rev_dict_updated[i] for i in resultList], "ratings":finalScores}


    #Store event that user clicked on the object
    def store_clicks(self, action, params):
        par_array = params.split("::")
        methodName = par_array[0]
        if methodName == "bandit":
            baseMethodName = par_array[1]
            import sqlite3
            conn = sqlite3.connect('hyperparamsDb.sqlite')
            c = conn.cursor()
            sql_params = (baseMethodName,)
            c.execute("UPDATE banditParams set r=r+1 where method=? ", sql_params)
            conn.commit()
            conn.close()

        elif methodName == "dhondt":
            self.update_dhondt_params(action, par_array)

    #store event that item was displayed to the user
    def store_views(self, action, params):
        par_array = params.split("::")
        methodName = par_array[0]
        if methodName == "bandit":
            baseMethodName = par_array[1]
            import sqlite3
            conn = sqlite3.connect('hyperparamsDb.sqlite')
            c = conn.cursor()
            sql_params = (baseMethodName,)
            c.execute("UPDATE banditParams set n=n+1 where method=? ", sql_params)
            conn.commit()
            conn.close()
        elif methodName == "dhondt":
            self.update_dhondt_params(action, par_array)

    def update_dhondt_params(self, action, par_array):
        # TODO: maybe store learning rates to a database?
        learningRateClicks = 0.03
        learningRateViews = learningRateClicks / 250
        maxVotesConst = 0.99
        minVotesConst = 0.01
        
        # TODO: try to enclose all this into a single transaction?
        import sqlite3
        conn = sqlite3.connect('hyperparamsDb.sqlite')
        c = conn.cursor()
        methods_list = []
        r_c_i_list = []
        # TODO: perform update for clicked item
        # gradient descend step maximizing selected item's relevance
        # alpha_i denotes votes for party_i; r_c_i denotes preference of party_i to selected candicate c
        # derivation by alpha_i(i.e., delta_alpha_i)  := r_c_i - sum_{forall j!=i}(r_c_j)
        # update step for alpha_i: alpha_i = alpha_i + learningRateClicks * delta_alpha_i
        # linearly scale alphas to maintain sum-all-alphas == 1

        # prepare data
        for baseMethod in par_array[1:]:
            (bmName, r_c_i) = baseMethod.split(":")
            methods_list.append(bmName)
            r_c_i_list.append(r_c_i)

        relevanceData = pd.Series(r_c_i_list, methods_list, name="candidate_relevance")
        relevanceData = relevanceData.astype(float)
        #print(relevanceData)
        origValues = pd.read_sql_query("SELECT * FROM dhondtParams;", conn)
        origValues.set_index("method", inplace=True)

        # update step for each method
        for m in relevanceData.index:
            relevance_this = relevanceData[m]
            relevance_others = relevanceData.sum() - relevance_this
            if action == "storeClicks":
                update_step = learningRateClicks * (relevance_this - relevance_others)
                pos_step = relevance_this
            elif action == "storeViews":
                update_step = -1 * learningRateViews * (relevance_this - relevance_others)
                pos_step = 0

            origValues.votes.loc[m] = origValues.votes.loc[m] + update_step
            
            #Apply constraints on maximal and minimal volumes of votes
            if origValues.votes.loc[m] < minVotesConst:
                origValues.votes.loc[m] = minVotesConst
            elif origValues.votes.loc[m] > maxVotesConst:
                origValues.votes.loc[m] = maxVotesConst
              
            origValues.click_share.loc[m] = origValues.click_share.loc[m] + pos_step

        # linearly normalizing to unit sum of votes
        origValues.votes = origValues.votes / origValues.votes.sum()

        sql_params = list(zip(origValues.votes, origValues.click_share, origValues.votes.index))
        c.executemany("UPDATE dhondtParams set votes=?, click_share=? where method=? ", sql_params)
        conn.commit()
        conn.close()


print('starting recommender...')
recommender = Recommender()

# HTTPRequestHandler class
class Reveal_HTTPServer_RequestHandler(BaseHTTPRequestHandler):

  def parse_POST(self):
        ctype, pdict = parse_header(self.headers['content-type'])
        if ctype == 'multipart/form-data':
            postvars = parse_multipart(self.rfile, pdict)
        elif ctype == 'application/x-www-form-urlencoded':
            length = int(self.headers['content-length'])
            postvars = parse_qs(
                self.rfile.read(length),
                keep_blank_values=1)
        else:
            postvars = {}
        return postvars

  def do_GET(self):
      self.send_response(200)
      # Send headers
      self.send_header('Content-type', 'text/html')
      self.end_headers()
      params = parse_qs(urlparse(self.path).query)
      #print(params.get("uid", [""]))
      #print(params.get("action", [""]))
      #print(params.get("par", [""]))

      if params.get("action", "") != "":
          print("PARAM updates")
          print(params.get("par", "none so far"))

          action = params.get("action", [""])[0]
          if action == "storeClicks":
              methodParams = params.get("par", [""])[0]
              recommender.store_clicks(action, methodParams)
          elif action == "storeViews":
              methodParams = params.get("par", [""])[0]
              recommender.store_views(action, methodParams)
          
      if params.get("uid", "") != "":  # it is a valid request, no favicon etc
          print("GET RECS request")
          # postvars =
          # Send response status code
          #try:
          if True:
              uid = int(params["uid"][0])
              allowed_oids = [int(i) for i in params.get("allowed_oids", [""])[0].split(",") if len(i) > 0]
              visited_oids = [int(i) for i in params.get("visited_oids", [""])[0].split(",") if
                              len(i) > 0]  # from oldest to newest
              visits_datetime = params.get("visits_datetime", [""])[0].split(",")  # from oldest to newest
              #print(visited_oids)
              #print(visits_datetime)
              #print(len(allowed_oids))

              now = datetime.datetime.now()
              # now = datetime.datetime(2018, 7, 20, 00, 00) #maybe put actual now
              visits_logDays = [
                  1 / (math.log(max([(now - datetime.datetime.strptime(i, '%Y-%m-%d %H:%M:%S')).days, 1])) + 0.1)
                  for i in visits_datetime]
              k = len(recommender.algorithm_descriptions)

              algorithmVariant = uid % k
              alg = recommender.algorithm_descriptions[algorithmVariant]
              #print(len(rev_dict))
              results = recommender.recommend(algorithmVariant, visited_oids, visits_logDays, allowed_oids)
              #results = recommender.recommend(model, dictionary, rev_dict, visited_oids, visits_logDays, alg["name"],
              #                                alg["aggregation"], alg["diversity"], alg["novelty"], allowed_oids)
              #print(results)
              resultsTxt = ",".join([str(i[0])+";"+str(i[1]) for i in results])

          #except:
          #    resultsTxt = "error"
          #    print ("error")
          message = resultsTxt
          self.send_response(200)
          # Send headers
          self.send_header('Content-type', 'text/html')
          self.end_headers()

          # store the query and response to the logfile
          # Send message back to client
          # response: coma separated top-20 recommended objects
          # on error return "error"
          with open("log.txt", "a") as f:
              f.write("{}::{}::{}::{}::{}\n".format(now, uid, visited_oids, (alg["name"], alg["params"], alg["aggregation"], alg["diversity"], alg["novelty"]), resultsTxt))
          # print(message)
          print(datetime.datetime.now() - now)
          # Write content as utf-8 data
          self.wfile.write(bytes(message, "utf8"))
          
      
      return

  def do_POST(self):
     #params = self.parse_POST()
     params = parse_qs(urlparse(self.path).query)
     if params.get("uid", "") != "": #it is a valid request, no favicon etc
        print("POST request")
        #postvars =
        # Send response status code
        try:
        #if True:
            uid = int(params["uid"][0])
            allowed_oids = [int(i) for i in params.get("allowed_oids",[""])[0].split(",") if len(i) > 0]
            visited_oids = [int(i) for i in params.get("visited_oids",[""])[0].split(",") if len(i) > 0] #from oldest to newest
            visits_datetime = params.get("visits_datetime",[""])[0].split(",")  # from oldest to newest

            now = datetime.datetime.now()
            #now = datetime.datetime(2018, 7, 20, 00, 00) #maybe put actual now
            visits_logDays = [1 / (math.log(max([(now - datetime.datetime.strptime(i, '%Y-%m-%d %H:%M:%S')).days, 1])) + 0.1)
                      for i in visits_datetime]
            k = len(recommender.algorithm_descriptions)
            algorithmVariant = uid % k
            alg = recommender.algorithm_descriptions[algorithmVariant]
            model = recommender.alg_models[(alg["name"], alg["params"])][0]
            dictionary = recommender.alg_models[(alg["name"], alg["params"])][1]
            rev_dict = recommender.alg_models[(alg["name"], alg["params"])][2]
            results = recommender.recommend(model, dictionary, rev_dict, visited_oids, visits_logDays, alg["name"], alg["aggregation"], alg["diversity"], alg["novelty"], allowed_oids)
            resultsTxt = ",".join([str(i) for i in results])

        except:
            resultsTxt = "error"
        message = resultsTxt
        self.send_response(200)
        # Send headers
        self.send_header('Content-type','text/html')
        self.end_headers()

        # store the query and response to the logfile
        # Send message back to client
        # response: coma separated top-20 recommended objects
        # on error return "error"
        with open("log.txt", "a") as f:
            f.write("{};{};{};{};{}\n".format(now, uid,visited_oids, (alg["name"], alg["params"], alg["aggregation"], alg["diversity"], alg["novelty"]), resultsTxt))
        #print(message)
        print(datetime.datetime.now() - now)
        # Write content as utf-8 data
        self.wfile.write(bytes(message, "utf8"))
     return

def run():
    print('starting server...')  # Server settings
    # Choose port 8080, for port 80, which is normally used for a http server, you need root access
    server_address = ('', 50000)
    httpd = HTTPServer(server_address, Reveal_HTTPServer_RequestHandler)
    print('running server...')
    httpd.serve_forever()

run()