// External adapter; original headers and sorting/search algorithms remain unchanged.
#include "iRG_search_multi.h"
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <unordered_set>
int main(int argc,char**argv){try{
 if(argc!=4)throw std::runtime_error("usage: bench_irange CONVERTED_DIR INDEX_FILE CSV");
 namespace fs=std::filesystem;fs::path dir=argv[1];if(fs::exists(argv[3]))throw std::runtime_error("output exists");
 iRangeGraph_multi::DataLoader store;store.query_K=20;store.LoadQuery((dir/"queries.bin").string());store.LoadData((dir/"vectors.bin").string());store.LoadAttribute((dir/"attr0.bin").string());store.LoadAttribute((dir/"attr1.bin").string());
 for(auto suite:{"fixedcard"})for(auto tag:{"r0050"}){
  std::string pre=std::string(suite)+"_"+tag;std::ifstream ranges(dir/(pre+"_ranges.bin"),std::ios::binary),gt(dir/(pre+"_gt.bin"),std::ios::binary);int n,k;gt.read((char*)&n,4);gt.read((char*)&k,4);if(n!=1000||k!=20||store.query_nb!=n)throw std::runtime_error("shape mismatch");
  for(int i=0;i<n;++i){int bounds[4];ranges.read((char*)bounds,16);store.query_range[pre].push_back({{{bounds[0],bounds[1]},{bounds[2],bounds[3]}}});std::vector<int> ids(20);gt.read((char*)ids.data(),80);gt.seekg(80,std::ios::cur);store.ground_truth[pre].push_back(ids);}
  if(!ranges||!gt)throw std::runtime_error("truncated data");
 }
 store.Sort_by_Attr(0);iRangeGraph_multi::iRangeGraph_Search_Multi<float> index(argv[2],&store,32);index.setprob();
 std::ofstream raw(std::string(argv[3])+".ids.tsv");raw<<"suite\ttag\tef\trepeat\tquery\tids\n";
 std::ofstream out(argv[3]);out<<"suite,tag,queries,k,ef,repeat,recall,qps,returned,validated\n";
 for(auto suite:{"fixedcard"})for(auto tag:{"r0050"}){
  std::string pre=std::string(suite)+"_"+tag;
  for(int rep=1;rep<=3;++rep)for(int ef:{20,32,64,128,256,512,1024,2048}){
   for(int i=0;i<20;++i){auto b=store.mapped_queryrange[pre][i];if(b.first>b.second)continue;auto nodes=index.tree->range_filter(index.tree->root,b.first,b.second);index.TopDown_search(store.query_points[i].data(),ef,20,b.first,b.second,32,store.query_range[pre][i].attr_constraints,nodes);}
   std::vector<std::vector<iRangeGraph_multi::PFI>> got(1000);auto start=std::chrono::steady_clock::now();
   for(int i=0;i<1000;++i){auto bounds=store.mapped_queryrange[pre][i];if(bounds.first>bounds.second)continue;auto nodes=index.tree->range_filter(index.tree->root,bounds.first,bounds.second);auto heap=index.TopDown_search(store.query_points[i].data(),ef,20,bounds.first,bounds.second,32,store.query_range[pre][i].attr_constraints,nodes);got[i].reserve(heap.size());while(!heap.empty()){got[i].push_back(heap.top());heap.pop();}}
   double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();size_t hits=0,returned=0;
   for(int i=0;i<1000;++i){std::unordered_set<int> seen;raw<<suite<<'\t'<<tag<<'\t'<<ef<<'\t'<<rep<<'\t'<<i<<'\t';for(auto pair:got[i]){int id=pair.second;raw<<id<<' ';if(id<0||id>=store.data_nb||!seen.insert(id).second)throw std::runtime_error("invalid or duplicate ID");for(int d=0;d<2;++d){auto b=store.query_range[pre][i].attr_constraints[d];if(store.attributes[id][d]<b.first||store.attributes[id][d]>b.second)throw std::runtime_error("out-of-filter output");}auto&gt=store.ground_truth[pre][i];hits+=std::find(gt.begin(),gt.end(),id)!=gt.end();++returned;}raw<<'\n';if(seen.size()>20)throw std::runtime_error("over k");}
   out<<suite<<','<<tag<<",1000,20,"<<ef<<','<<rep<<','<<std::setprecision(12)<<hits/20000.0<<','<<1000/sec<<','<<returned<<",1\n";out.flush();
  }
 }
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
