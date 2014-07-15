from elasticsearch import Elasticsearch
import numpy as np
import matplotlib.pyplot as pl
from sklearn.decomposition import PCA, FactorAnalysis, ProbabilisticPCA, TruncatedSVD
from sklearn.manifold import MDS
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from datetime import date, datetime, time, timedelta

today = datetime.now()
start_date = datetime(2013, 5, 1)
td = timedelta(days=15)

final_csv = ""
datas = {'date':[]}
while start_date + timedelta(days=30) < datetime(2014, 6, 1):
    end_date = start_date + timedelta(days=30)
    start_date_str = "%04d" % start_date.year + "%02d" % start_date.month + "%02d" % start_date.day
    end_date_str = "%04d" % end_date.year + "%02d" % end_date.month + "%02d" % end_date.day
    datas['date'].append(start_date_str)
    es = Elasticsearch("192.168.0.104")


    def retrieve_votazioni():
        query = {
          "query": {
            "bool": {
              "must": [
                {
                  "term": {
                    "votazione.source": "camera"
                  }
                },
                {
                  "term": {
                    "votazione.palese": True
                  }
                },
                {"range":{"votazione.date":{"from":start_date_str,"to":end_date_str}}}
              ],
              "must_not":[
                  #{
                      #"query_string":
                      # {
                      #     "default_field":"votazione.name",
                      #     "query":"ordine"
                      # }
                  #}
              ],
              "should": []
            }
          },
        }

        votes = es.search(index="votazioni_index5", body=query, size=10000)
        return [x for x in votes['hits']['hits']][1:]

    votazioni = retrieve_votazioni()

    votes = {}

    color_map = {"IL POPOLO DELLA LIBERTA' - BERLUSCONI PRESIDENTE":'blue',
                 "FORZA ITALIA - IL POPOLO DELLA LIBERTA' XVII LEGISLATURA":'blue',
                 "LEGA NORD E AUTONOMIE":'green',
                 "PARTITO DEMOCRATICO":'orange',
                 "SCELTA CIVICA PER L'ITALIA":'gray',
                 "MOVIMENTO 5 STELLE":'yellow',
                 "SINISTRA ECOLOGIA LIBERTA'":'red'}

    totale_votazioni = 0
    votes_per_person = {}
    politicians_by_group = {"FRATELLI D'ITALIA":set([]),"MISTO - MAIE - MOVIMENTO ASSOCIATIVO ITALIANI ALL'ESTERO": set([]), "SCELTA CIVICA PER L'ITALIA": set([]), "PER L'ITALIA": set([]), 'NUOVO CENTRODESTRA': set([]), "IL POPOLO DELLA LIBERTA' - BERLUSCONI PRESIDENTE": set([]), "FRATELLI D'ITALIA - ALLEANZA NAZIONALE": set([]), 'MOVIMENTO 5 STELLE': set([]), 'MISTO - NON ISCRITTO AD ALCUNA COMPONENTE POLITICA': set([]), "FORZA ITALIA - IL POPOLO DELLA LIBERTA' - BERLUSCONI PRESIDENTE": set([]), "FORZA ITALIA - IL POPOLO DELLA LIBERTA' XVII LEGISLATURA": set([]), "SINISTRA ECOLOGIA LIBERTA'": set([]), 'MISTO - CENTRO DEMOCRATICO': set([]), "MISTO - MAIE - MOVIMENTO ASSOCIATIVO ITALIANI ALL'ESTERO - ALLEANZA PER L'ITALIA (API)": set([]), "MISTO - PARTITO SOCIALISTA ITALIANO (PSI) - LIBERALI PER L'ITALIA (PLI)": set([]), 'PARTITO DEMOCRATICO': set([]), 'MISTO - MINORANZE LINGUISTICHE': set([]), 'LEGA NORD E AUTONOMIE': set([])}

    groups_by_politician = {}

    for votazione in votazioni:
        totale_votazioni += 1
        data = votazione['_source']['person_data']
        for person_vote in data:
            person = person_vote['person']
            #if person_vote['gruppo'] not in politicians_by_group:
            #    continue
            politicians_by_group[person_vote['gruppo']].add(person)
            groups_by_politician[person] = person_vote['gruppo']
            vote = person_vote['result']
            #if vote == "Ha votato":
            #    continue
            if vote not in ["Assente", "In missione"]:
                if person in votes_per_person:
                    votes_per_person[person] += 1
                else:
                    votes_per_person[person] = 1

            if vote == "Favorevole":
                vote = 1 #"Favorevole"
            elif vote == "Contrario":
                vote = -1 #"Contrario"
            elif vote == "Assente" or vote == "In missione":
                vote = 0.0
            else:
                vote = 0.0 #"Nothing"
            if person in votes:
                votes[person].append(vote)
            else:
                votes[person] = [vote]

    lengths = {}

    for i in [i for i in votes.values()]:
        if len(i) in lengths:
            lengths[len(i)] += 1
        else:
            lengths[len(i)] = 1

    if len(lengths) > 1:
        #print "Warning, different voting lengths - (pruning):", lengths
        most_common_length = max(lengths.items(), key = lambda x: x[1])[0]
        for i in votes.keys():
            if len(votes[i]) != most_common_length:
                del votes[i]

    for person in votes.keys():
        if float(votes_per_person.get(person, 0))/float(totale_votazioni) < .4:
            del votes[person]

    people = votes.keys()
    corr_matrix = np.corrcoef(np.array([i for i in votes.values()]))
    corr_matrix = np.nan_to_num(corr_matrix)

    models = [('mds', MDS(n_components=2)), ('LSA', TruncatedSVD(n_components=2)), ('FA', FactorAnalysis(n_components=2)),('pca', PCA(n_components=2))]
    plotno = 221
    for model_name, model in models:
        pl.subplot(plotno)
        pl.title(model_name)
        plotno += 1

        reduced_matrix = model.fit_transform(corr_matrix)

        #np.savetxt("pca_matrix.csv", reduced_matrix, delimiter=",")
        x = reduced_matrix[:,0]
        y = reduced_matrix[:,1]
        for i, xi in enumerate(x):
            yi = y[i]
            person = people[i]
            group = groups_by_politician[people[i]]
            pl.plot(xi, yi, 'go', color=color_map.get(group,"gray"))

        distances = squareform(pdist(reduced_matrix))

        #print start_date_str + " to " + end_date_str
        #v1, v2 = 0,0#pca.explained_variance_ratio_

        #final_csv += "%s,%s,%s,%s," % (v1, v2, v1+v2, start_date_str)

        data = []
        for party, group in politicians_by_group.items():
            elements = []
            for politician1 in group:
                if not politician1 in people:
                    continue
                i = people.index(politician1)
                for politician2 in group:
                    if not politician2 in people or (politician1 == politician2):
                        continue
                    j = people.index(politician2)
                    elements.append(distances[i, j])
            f = float(sum(elements))/(len(elements) or 1)
            data.append(f)
            if party in datas:
                datas[party].append(f)
            else:
                datas[party] = [f]

        #final_csv += ",".join(map(str, data)) + "\n"

        start_date += td



    pl.show()




csv_header = "v1,v2,v12,date,"
csv_header += ",".join(politicians_by_group.keys())+"\n"

with open("test.csv", "w") as f:
    f.write(csv_header+final_csv)

for party in datas.keys():
    if party != 'date':
        pl.plot(range(len(datas['date'])), list(datas[party]), label=party)



pl.legend(fontsize='xx-small')
pl.show()