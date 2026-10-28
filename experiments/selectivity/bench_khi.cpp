// External measurement/validation adapter; all KHI algorithms are unchanged author code.
#include "ma_graph.h"
#include <filesystem>
int main(int argc,char**argv){try{
 if(argc!=4)throw std::runtime_error("usage: bench_khi CONVERTED_DIR INDEX_PREFIX OUTPUT_CSV");
 namespace fs=std::filesystem;fs::path dir=argv[1];if(fs::exists(argv[3]))throw std::runtime_error("output exists");
 Data data((dir/"meta.bin").string(),(dir/"vectors.bin").string());MAGraph index(&data);index.load(argv[2]);
 index.prefetch_lines=((data.d+7)/8*8*sizeof(float))>>4;
 std::ofstream raw(std::string(argv[3])+".ids.tsv");raw<<"suite\ttag\tef\trepeat\tquery\tids\n";
 std::ofstream out(argv[3]);out<<"suite,tag,queries,k,ef,repeat,recall,qps,returned,validated\n";
 for(auto suite:{"fixedcard"})for(auto tag:{"r0050"}){
  std::string pre=std::string(suite)+"_"+tag;Query q;q.load_queries((dir/"queries.bin").string());q.load_constraints((dir/(pre+"_constraints.bin")).string(),&data);q.load_answers((dir/(pre+"_gt.bin")).string());
  if(q.n_q!=1000)throw std::runtime_error("query count");
  for(int rep=1;rep<=3;++rep)for(int ef:{20,32,64,128,256,512,1024,2048}){
   for(int i=0;i<20;++i){auto nodes=index.range_filter(q.q_a[i],20);index.TopDown_nodeentries_search(nodes,q.q_v[i],ef,20,q.q_a[i],32);}
   std::vector<std::vector<PFI>> got(q.n_q);auto start=std::chrono::steady_clock::now();
   for(int i=0;i<q.n_q;++i){auto nodes=index.range_filter(q.q_a[i],20);auto heap=index.TopDown_nodeentries_search(nodes,q.q_v[i],ef,20,q.q_a[i],32);got[i].reserve(heap.size());while(!heap.empty()){got[i].push_back(heap.top());heap.pop();}}
   double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();size_t hits=0,returned=0;
   for(int i=0;i<q.n_q;++i){std::unordered_set<int> seen;raw<<suite<<'\t'<<tag<<'\t'<<ef<<'\t'<<rep<<'\t'<<i<<'\t';for(auto pair:got[i]){int id=pair.second;raw<<id<<' ';if(id<0||id>=data.n||!seen.insert(id).second||!index.check_constraint(id,q.q_a[i]))throw std::runtime_error("invalid/duplicate/out-of-filter output");hits+=std::find(q.knn_id[i].begin(),q.knn_id[i].begin()+20,id)!=q.knn_id[i].begin()+20;++returned;}raw<<'\n';if(seen.size()>20)throw std::runtime_error("over k");}
   out<<suite<<','<<tag<<",1000,20,"<<ef<<','<<rep<<','<<std::setprecision(12)<<hits/20000.0<<','<<1000/sec<<','<<returned<<",1\n";out.flush();
  }
 }
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
