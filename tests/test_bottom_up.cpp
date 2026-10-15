#include "IndexCube.h"
#include "config.h"
#include <filesystem>
#include <random>
#include <set>
#include <iostream>
#include <stdexcept>
namespace fs = std::filesystem;
void require(bool x, const char* why) { if (!x) throw std::runtime_error(why); }
void check_graph(IndexCube& index, size_t n) {
    size_t layer=0;
    while (auto* g=index.get_hnsw_index(layer++)) {
        require(g->cur_element_count==n,"point count");
        std::set<unsigned> labels;
        for (unsigned i=0;i<n;++i) {
            require(labels.insert(g->getExternalLabel(i)).second,"duplicate label");
            require(g->label_lookup_.at(g->getExternalLabel(i))==i,"label map");
            auto ll=g->get_linklist0(i); auto e=(unsigned*)(ll+1);
            require(g->getListCount(ll)<=g->maxM0_,"degree cap");
            std::set<unsigned> seen;
            for (unsigned j=0;j<g->getListCount(ll);++j) {
                require(e[j]<n && e[j]!=i,"local ID");
                require(seen.insert(e[j]).second,"duplicate edge");
                require(g->getCubeId(e[j])==g->getCubeId(i),"local edge crosses cube");
            }
            auto c=g->getCrossEdges(i);
            for (unsigned j=0;j<g->Meta_dim_*2*g->Cross_edge_counts_;++j) if(c[j]) {
                require(c[j]<n,"cross ID");
                auto& adj=g->adjacent_cube_ids_[g->getCubeId(i)];
                require(std::find(adj.begin(),adj.end(),g->getCubeId(c[j]))!=adj.end(),"nonadjacent cross edge");
            }
        }
        for (auto ep:g->cube_entry_points_) if(ep!=unsigned(-1)) require(ep<n,"entry ID");
        if(layer>1) {auto* prev=index.get_hnsw_index(layer-2);
            for(unsigned i=0;i<n;++i) require(prev->getExternalLabel(i)==g->getExternalLabel(i),"cross-layer ID mapping");}
    }
}
int main(int argc,char** argv) {
 try {
    fs::path dir=argc>1?argv[1]:"/tmp/cube-bottom-up-test";fs::create_directories(dir);
    constexpr size_t n=4096,d=8;
    std::mt19937 rng(771);std::uniform_real_distribution<float> unif(0,1);
    std::vector<float> vectors(n*d);for(auto&v:vectors)v=unif(rng);
    auto vf=(dir/"base.fvecs").string();{std::ofstream o(vf,std::ios::binary);int dd=d;for(size_t i=0;i<n;++i){o.write((char*)&dd,4);o.write((char*)&vectors[i*d],d*4);}}
    for (size_t m:{2,3,4}) for(size_t cross:{0,2}) {
        std::vector<float> meta(n*m);for(auto&v:meta)v=unif(rng);
        // Include degenerate dimension and nonuniform occupancy.
        if(m==3) for(size_t i=0;i<n;++i)meta[i*m+2]=.5f;
        auto mf=(dir/"meta.bin").string();{std::ofstream o(mf,std::ios::binary);o.write((char*)&n,8);o.write((char*)&m,8);o.write((char*)meta.data(),meta.size()*4);}
        IndexCube a(3,16,100,cross,8);a.build_index(vf.data(),mf.data());check_graph(a,n);
        auto ix=(dir/("index"+std::to_string(m)+"_"+std::to_string(cross))).string();a.save_index(ix);
        IndexCube b;require(b.load_index(ix,vf.data()),"load");check_graph(b,n);
        a.set_global_ef(n);b.set_global_ef(n);
        for(unsigned q=0;q<16;++q) {
            BoundingBox box(m);for(size_t j=0;j<m;++j){box.min_bounds[j]=q<8?0:.2f;box.max_bounds[j]=q<8?1:.8f;}
            auto ra=a.fly_search(&vectors[q*d],20,&box),rb=b.fly_search(&vectors[q*d],20,&box);
            require(ra.size()==rb.size(),"save/load result count");std::set<unsigned> got;
            while(!ra.empty()){require(ra.top()==rb.top(),"save/load query equality");rb.pop();auto id=ra.top().second;ra.pop();require(got.insert(id).second,"duplicate answer");for(size_t j=0;j<m;++j)require(meta[id*m+j]>=box.min_bounds[j]&&meta[id*m+j]<=box.max_bounds[j],"predicate violation");}
            // Exact recall is expected for connected cross-enabled synthetic graphs
            // at exhaustive search width; cross0 is intentionally disconnected.
            if(cross) {
                std::vector<std::pair<float,unsigned>> exact;
                for(unsigned id=0;id<n;++id){bool ok=true;for(size_t j=0;j<m;++j)ok &= meta[id*m+j]>=box.min_bounds[j]&&meta[id*m+j]<=box.max_bounds[j];if(ok){float dist=0;for(size_t j=0;j<d;++j){float z=vectors[id*d+j]-vectors[q*d+j];dist+=z*z;}exact.emplace_back(dist,id);}}
                std::sort(exact.begin(),exact.end());size_t hits=0;for(size_t k=0;k<std::min(size_t(20),exact.size());++k)hits+=got.count(exact[k].second);
                require(hits==std::min(size_t(20),exact.size()),"exhaustive recall");
            }
        }
        std::cout<<"PASS dims="<<m<<" cross="<<cross<<std::endl;
    }
    std::cout<<"ALL_TESTS_PASS"<<std::endl;
 }catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}
}
