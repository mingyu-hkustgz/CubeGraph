#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "IndexRTreePartition.h"
#include "IndexCube.h"
#include "hnsw-static.h"
#include "config.h"

#include <memory>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

BoundingBox make_bbox_from_vectors(const std::vector<float> &min_bounds,
                                   const std::vector<float> &max_bounds) {
    if (min_bounds.size() != max_bounds.size()) {
        throw std::invalid_argument("min_bounds and max_bounds must have the same dimension");
    }

    BoundingBox bbox(min_bounds.size());
    bbox.min_bounds = min_bounds;
    bbox.max_bounds = max_bounds;
    return bbox;
}

std::vector<std::pair<float, hnswlib::labeltype>>
queue_to_sorted_vector(std::priority_queue<std::pair<float, hnswlib::labeltype>> pq) {
    std::vector<std::pair<float, hnswlib::labeltype>> out;
    out.reserve(pq.size());

    while (!pq.empty()) {
        out.push_back(pq.top());
        pq.pop();
    }

    std::reverse(out.begin(), out.end());
    return out;
}

std::vector<std::pair<float, hnswlib::labeltype>> search_with_filter_bounds(
        IndexRTreePartition &index,
        py::array_t<float, py::array::c_style | py::array::forcecast> query,
        size_t k,
        const std::vector<float> &min_bounds,
        const std::vector<float> &max_bounds) {
    if (query.ndim() != 1) {
        throw std::invalid_argument("query must be a 1D float array");
    }

    BoundingBox bbox = make_bbox_from_vectors(min_bounds, max_bounds);
    auto result = index.search(query.data(), k, bbox);
    return queue_to_sorted_vector(std::move(result));
}

std::vector<std::pair<float, hnswlib::labeltype>> search_with_bbox(
        IndexRTreePartition &index,
        py::array_t<float, py::array::c_style | py::array::forcecast> query,
        size_t k,
        const BoundingBox &bbox) {
    if (query.ndim() != 1) {
        throw std::invalid_argument("query must be a 1D float array");
    }

    auto result = index.search(query.data(), k, bbox);
    return queue_to_sorted_vector(std::move(result));
}

void build_index_wrapper(IndexRTreePartition &index,
                         const std::string &data_file,
                         const std::string &metadata_file) {
    index.build_index(const_cast<char*>(data_file.c_str()), const_cast<char*>(metadata_file.c_str()));
}

void load_metadata_wrapper(IndexRTreePartition &index, const std::string &metadata_file) {
    index.load_metadata(metadata_file.c_str());
}

void cube_load_metadata_wrapper(IndexCube &index, const std::string &metadata_file) {
    index.load_metadata(const_cast<char *>(metadata_file.c_str()));
}

void cube_build_index_wrapper(IndexCube &index,
                              const std::string &data_file,
                              const std::string &metadata_file) {
    index.build_index(const_cast<char *>(data_file.c_str()), const_cast<char *>(metadata_file.c_str()));
}

bool cube_load_index_wrapper(IndexCube &index,
                             const std::string &index_path,
                             const std::string &data_file) {
    return index.load_index(index_path, const_cast<char *>(data_file.c_str()));
}

std::vector<std::pair<float, hnswlib::labeltype>> cube_fly_search_bbox(
        IndexCube &index,
        py::array_t<float, py::array::c_style | py::array::forcecast> query,
        size_t k,
        const BoundingBox &bbox) {
    if (query.ndim() != 1) {
        throw std::invalid_argument("query must be a 1D float array");
    }
    auto result = index.fly_search(query.data(), k, &bbox);
    return queue_to_sorted_vector(std::move(result));
}

std::vector<std::pair<float, hnswlib::labeltype>> cube_predetermined_search_bbox(
        IndexCube &index,
        py::array_t<float, py::array::c_style | py::array::forcecast> query,
        size_t k,
        const BoundingBox &bbox) {
    if (query.ndim() != 1) {
        throw std::invalid_argument("query must be a 1D float array");
    }
    auto result = index.predetermined_search(query.data(), k, &bbox);
    return queue_to_sorted_vector(std::move(result));
}

std::vector<std::pair<float, hnswlib::labeltype>> cube_fly_search_radius(
        IndexCube &index,
        py::array_t<float, py::array::c_style | py::array::forcecast> query,
        size_t k,
        const RadiusFilterParams &radius) {
    if (query.ndim() != 1) {
        throw std::invalid_argument("query must be a 1D float array");
    }
    auto result = index.fly_search(query.data(), k, &radius);
    return queue_to_sorted_vector(std::move(result));
}

std::vector<std::pair<float, hnswlib::labeltype>> cube_predetermined_search_radius(
        IndexCube &index,
        py::array_t<float, py::array::c_style | py::array::forcecast> query,
        size_t k,
        const RadiusFilterParams &radius) {
    if (query.ndim() != 1) {
        throw std::invalid_argument("query must be a 1D float array");
    }
    auto result = index.predetermined_search(query.data(), k, &radius);
    return queue_to_sorted_vector(std::move(result));
}

std::vector<std::pair<float, hnswlib::labeltype>> cube_fly_search_polygon(
        IndexCube &index,
        py::array_t<float, py::array::c_style | py::array::forcecast> query,
        size_t k,
        const PolygonFilterParams &polygon) {
    if (query.ndim() != 1) {
        throw std::invalid_argument("query must be a 1D float array");
    }
    auto result = index.fly_search(query.data(), k, &polygon);
    return queue_to_sorted_vector(std::move(result));
}

std::vector<std::pair<float, hnswlib::labeltype>> cube_predetermined_search_polygon(
        IndexCube &index,
        py::array_t<float, py::array::c_style | py::array::forcecast> query,
        size_t k,
        const PolygonFilterParams &polygon) {
    if (query.ndim() != 1) {
        throw std::invalid_argument("query must be a 1D float array");
    }
    auto result = index.predetermined_search(query.data(), k, &polygon);
    return queue_to_sorted_vector(std::move(result));
}

std::vector<std::pair<float, hnswlib::labeltype>> cube_fly_search_composite(
        IndexCube &index,
        py::array_t<float, py::array::c_style | py::array::forcecast> query,
        size_t k,
        const CompositeFilterParams &params) {
    if (query.ndim() != 1) {
        throw std::invalid_argument("query must be a 1D float array");
    }
    auto result = index.fly_search(query.data(), k, &params);
    return queue_to_sorted_vector(std::move(result));
}

std::vector<std::pair<float, hnswlib::labeltype>> cube_predetermined_search_composite(
        IndexCube &index,
        py::array_t<float, py::array::c_style | py::array::forcecast> query,
        size_t k,
        const CompositeFilterParams &params) {
    if (query.ndim() != 1) {
        throw std::invalid_argument("query must be a 1D float array");
    }
    auto result = index.predetermined_search(query.data(), k, &params);
    return queue_to_sorted_vector(std::move(result));
}

class BBoxMetaFilter : public hnswlib::MetaFilterFunctor {
public:
    explicit BBoxMetaFilter(const BoundingBox &bbox) : bbox_(bbox) {}
    bool operator()(hnswlib::metatype *meta) override {
        for (size_t d = 0; d < bbox_.min_bounds.size(); d++) {
            float v = meta[d];
            if (v < bbox_.min_bounds[d] || v > bbox_.max_bounds[d]) {
                return false;
            }
        }
        return true;
    }
private:
    const BoundingBox &bbox_;
};

std::vector<std::vector<float>> load_metadata_file(const std::string &path, size_t &n, size_t &d) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open metadata file: " + path);
    }
    in.read(reinterpret_cast<char *>(&n), sizeof(size_t));
    in.read(reinterpret_cast<char *>(&d), sizeof(size_t));
    std::vector<std::vector<float>> metadata(n, std::vector<float>(d));
    for (size_t i = 0; i < n; i++) {
        in.read(reinterpret_cast<char *>(metadata[i].data()), d * sizeof(float));
    }
    return metadata;
}

class PostSearchIndex {
public:
    PostSearchIndex(size_t M = 16, size_t ef_construction = 200)
            : M_(M), ef_construction_(ef_construction), meta_dim_(0) {}

    void load_metadata(const std::string &metadata_file) {
        size_t n = 0;
        size_t d = 0;
        metadata_ = load_metadata_file(metadata_file, n, d);
        meta_dim_ = d;
    }

    void build_index(const std::string &data_file, const std::string &metadata_file) {
        load_metadata(metadata_file);

        data_matrix_ = std::make_unique<Matrix<float>>(const_cast<char *>(data_file.c_str()));
        hnswlib::HierarchicalNSWStatic<float>::static_base_data_ = (char *) data_matrix_->data;
        space_ = std::make_unique<hnswlib::L2Space>(data_matrix_->d);
        index_ = std::make_unique<hnswlib::HierarchicalNSWStatic<float>>(
                space_.get(), data_matrix_->n, M_, ef_construction_);

        index_->set_meta_dim(meta_dim_);
        index_->get_metadata() = metadata_;

#pragma omp parallel for schedule(dynamic, 144)
        for (int i = 0; i < data_matrix_->n; i++) {
            index_->addPoint(data_matrix_->data + (size_t) i * (size_t) data_matrix_->d, i);
        }
    }

    void load_index(const std::string &index_file,
                    const std::string &data_file,
                    const std::string &metadata_file = "") {
        data_matrix_ = std::make_unique<Matrix<float>>(const_cast<char *>(data_file.c_str()));
        hnswlib::HierarchicalNSWStatic<float>::static_base_data_ = (char *) data_matrix_->data;
        space_ = std::make_unique<hnswlib::L2Space>(data_matrix_->d);
        index_ = std::make_unique<hnswlib::HierarchicalNSWStatic<float>>(space_.get(), index_file);

        if (!metadata_file.empty()) {
            load_metadata(metadata_file);
            index_->set_meta_dim(meta_dim_);
            index_->get_metadata() = metadata_;
        } else {
            meta_dim_ = index_->get_meta_dim();
            metadata_ = index_->get_metadata();
        }
    }

    void save_index(const std::string &index_file) {
        ensure_index();
        index_->saveIndex(index_file);
    }

    void set_ef(size_t ef) {
        ensure_index();
        index_->setEf(ef);
    }

    size_t get_meta_dim() const { return meta_dim_; }
    size_t get_num_vectors() const {
        if (!data_matrix_) return 0;
        return data_matrix_->n;
    }
    const std::vector<std::vector<float>> &get_metadata() const { return metadata_; }

    std::vector<std::pair<float, hnswlib::labeltype>> search(
            py::array_t<float, py::array::c_style | py::array::forcecast> query,
            size_t k,
            const BoundingBox &bbox) {
        ensure_index();
        if (query.ndim() != 1) {
            throw std::invalid_argument("query must be a 1D float array");
        }
        BBoxMetaFilter filter(bbox);
        auto result = index_->searchKnn(query.data(), k, &filter);
        return queue_to_sorted_vector(std::move(result));
    }

    std::vector<std::pair<float, hnswlib::labeltype>> search_with_bounds(
            py::array_t<float, py::array::c_style | py::array::forcecast> query,
            size_t k,
            const std::vector<float> &min_bounds,
            const std::vector<float> &max_bounds) {
        BoundingBox bbox = make_bbox_from_vectors(min_bounds, max_bounds);
        return search(query, k, bbox);
    }

private:
    void ensure_index() const {
        if (!index_) {
            throw std::runtime_error("Index is not built/loaded yet");
        }
    }

    size_t M_;
    size_t ef_construction_;
    size_t meta_dim_;
    std::vector<std::vector<float>> metadata_;

    std::unique_ptr<Matrix<float>> data_matrix_;
    std::unique_ptr<hnswlib::L2Space> space_;
    std::unique_ptr<hnswlib::HierarchicalNSWStatic<float>> index_;
};

}  // namespace

PYBIND11_MODULE(rtree_partition_py, m) {
    m.doc() = "Python bindings for IndexRTreePartition";

    py::class_<BoundingBox>(m, "BoundingBox")
        .def(py::init<>())
        .def(py::init<size_t>(), py::arg("dim"))
        .def_readwrite("min_bounds", &BoundingBox::min_bounds)
        .def_readwrite("max_bounds", &BoundingBox::max_bounds)
        .def("contains", &BoundingBox::contains, py::arg("point"))
        .def("overlaps", &BoundingBox::overlaps, py::arg("other"))
        .def("volume", &BoundingBox::volume);

    py::class_<Sphere>(m, "Sphere")
        .def(py::init<>())
        .def(py::init<size_t>(), py::arg("dim"))
        .def(py::init<const std::vector<float> &, float>(), py::arg("center"), py::arg("radius"))
        .def_readwrite("center", &Sphere::center)
        .def_readwrite("radius", &Sphere::radius)
        .def("overlaps", &Sphere::overlaps, py::arg("box"));

    py::class_<RadiusFilterParams>(m, "RadiusFilterParams")
        .def(py::init<>())
        .def(py::init<const std::vector<float> &, float, size_t>(),
             py::arg("center"), py::arg("radius"), py::arg("attr_dim"))
        .def_readwrite("sphere", &RadiusFilterParams::sphere)
        .def_readwrite("attr_dim", &RadiusFilterParams::attr_dim);

    py::class_<PolygonFilterParams>(m, "PolygonFilterParams")
        .def(py::init<>())
        .def(py::init<const std::vector<std::vector<float>> &, size_t>(),
             py::arg("vertices"), py::arg("attr_dim"))
        .def_readwrite("vertices", &PolygonFilterParams::vertices)
        .def_readwrite("num_vertices", &PolygonFilterParams::num_vertices)
        .def_readwrite("attr_dim", &PolygonFilterParams::attr_dim)
        .def("get_bbox", &PolygonFilterParams::get_bbox)
        .def("area", &PolygonFilterParams::area);

    py::enum_<CompositeFilterType>(m, "CompositeFilterType")
        .value("BBOX", CompositeFilterType::BBOX)
        .value("RADIUS", CompositeFilterType::RADIUS)
        .value("POLYGON", CompositeFilterType::POLYGON);

    py::class_<CompositeFilterSpec>(m, "CompositeFilterSpec")
        .def(py::init<>())
        .def_readwrite("use_bbox", &CompositeFilterSpec::use_bbox)
        .def_readwrite("use_radius", &CompositeFilterSpec::use_radius)
        .def_readwrite("use_polygon", &CompositeFilterSpec::use_polygon)
        .def_readwrite("bbox_negate", &CompositeFilterSpec::bbox_negate)
        .def_readwrite("radius_negate", &CompositeFilterSpec::radius_negate)
        .def_readwrite("polygon_negate", &CompositeFilterSpec::polygon_negate)
        .def_readwrite("primary_op", &CompositeFilterSpec::primary_op);

    py::class_<CompositeFilterParams>(m, "CompositeFilterParams")
        .def(py::init<>())
        .def_readwrite("bbox", &CompositeFilterParams::bbox)
        .def_readwrite("radius", &CompositeFilterParams::radius)
        .def_readwrite("polygon", &CompositeFilterParams::polygon)
        .def_readwrite("spec", &CompositeFilterParams::spec)
        .def_readwrite("attr_dim", &CompositeFilterParams::attr_dim);

    py::class_<IndexRTreePartition>(m, "IndexRTreePartition")
        .def(py::init<size_t, size_t, size_t, size_t, bool, size_t, size_t>(),
             py::arg("leaf_capacity") = 1000,
             py::arg("leaf_scan_threshold") = 100,
             py::arg("leaf_k_expand_factor") = 4,
             py::arg("leaf_ef_min") = 0,
             py::arg("leaf_use_meta_filter") = false,
             py::arg("M") = 16,
             py::arg("ef_construction") = 200)
        .def("set_ef", &IndexRTreePartition::set_ef, py::arg("ef"))
        .def("get_num_leaves", &IndexRTreePartition::get_num_leaves)
        .def("get_num_hnsw_leaves", &IndexRTreePartition::get_num_hnsw_leaves)
        .def("get_num_scan_leaves", &IndexRTreePartition::get_num_scan_leaves)
        .def("set_force_scan", &IndexRTreePartition::set_force_scan, py::arg("value"))
        .def("set_leaf_k_expand_factor", &IndexRTreePartition::set_leaf_k_expand_factor, py::arg("value"))
        .def("set_leaf_ef_min", &IndexRTreePartition::set_leaf_ef_min, py::arg("value"))
        .def("set_leaf_use_meta_filter", &IndexRTreePartition::set_leaf_use_meta_filter, py::arg("value"))
        .def("load_metadata", &load_metadata_wrapper, py::arg("metadata_file"))
        .def("build_index", &build_index_wrapper, py::arg("data_file"), py::arg("metadata_file"))
        .def("search",
             &search_with_bbox,
             py::arg("query"),
             py::arg("k"),
             py::arg("bbox"),
             "Search with a BoundingBox object. Returns [(distance, label), ...] sorted by distance.")
        .def("search_with_bounds",
             &search_with_filter_bounds,
             py::arg("query"),
             py::arg("k"),
             py::arg("min_bounds"),
             py::arg("max_bounds"),
             "Search with explicit min/max bounds. Returns [(distance, label), ...] sorted by distance.")
        .def("get_metadata", &IndexRTreePartition::get_metadata,
             py::return_value_policy::copy)
        .def("get_attr_dim", &IndexRTreePartition::get_attr_dim)
        .def("get_num_vectors", &IndexRTreePartition::get_num_vectors);

    py::class_<IndexCube>(m, "IndexCube")
        .def(py::init<size_t, size_t, size_t, size_t, size_t>(),
             py::arg("num_layers") = 3,
             py::arg("M") = 16,
             py::arg("ef_construction") = 200,
             py::arg("cross_edge_count") = 2,
             py::arg("min_points_per_cube") = 50)
        .def("set_global_ef", &IndexCube::set_global_ef, py::arg("ef"))
        .def("load_metadata", &cube_load_metadata_wrapper, py::arg("metadata_file"))
        .def("compute_global_bbox", &IndexCube::compute_global_bbox)
        .def("build_index", &cube_build_index_wrapper, py::arg("data_file"), py::arg("metadata_file"))
        .def("compute_adaptive_num_layers", &IndexCube::compute_adaptive_num_layers)
        .def("select_layer_with_bbox", [](const IndexCube &index, const BoundingBox &bbox) {
            return index.select_layer_with_BBox(&bbox);
        }, py::arg("bbox"))
        .def("find_cube_with_bbox", [](const IndexCube &index, size_t layer_id, const BoundingBox &bbox) {
            return index.find_cube_with_BBox(layer_id, &bbox);
        }, py::arg("layer_id"), py::arg("bbox"))
        .def("find_cube_list_with_bbox", [](const IndexCube &index, size_t layer_id, const BoundingBox &bbox) {
            return index.find_cube_list_with_BBox(layer_id, &bbox);
        }, py::arg("layer_id"), py::arg("bbox"))
        .def("select_layer_with_radius", [](const IndexCube &index, const RadiusFilterParams &radius) {
            return index.select_layer_with_Radius(&radius);
        }, py::arg("radius"))
        .def("find_cube_with_radius", [](const IndexCube &index, size_t layer_id, const RadiusFilterParams &radius) {
            return index.find_cube_with_Radius(layer_id, &radius);
        }, py::arg("layer_id"), py::arg("radius"))
        .def("find_cube_list_with_radius", [](const IndexCube &index, size_t layer_id, const RadiusFilterParams &radius) {
            return index.find_cube_list_with_Radius(layer_id, &radius);
        }, py::arg("layer_id"), py::arg("radius"))
        .def("select_layer_with_polygon", [](const IndexCube &index, const PolygonFilterParams &polygon) {
            return index.select_layer_with_Polygon(&polygon);
        }, py::arg("polygon"))
        .def("find_cube_with_polygon", [](const IndexCube &index, size_t layer_id, const PolygonFilterParams &polygon) {
            return index.find_cube_with_Polygon(layer_id, &polygon);
        }, py::arg("layer_id"), py::arg("polygon"))
        .def("find_cube_list_with_polygon", [](const IndexCube &index, size_t layer_id, const PolygonFilterParams &polygon) {
            return index.find_cube_list_with_Polygon(layer_id, &polygon);
        }, py::arg("layer_id"), py::arg("polygon"))
        .def("fly_search_bbox", &cube_fly_search_bbox, py::arg("query"), py::arg("k"), py::arg("bbox"))
        .def("predetermined_search_bbox", &cube_predetermined_search_bbox, py::arg("query"), py::arg("k"), py::arg("bbox"))
        .def("fly_search_radius", &cube_fly_search_radius, py::arg("query"), py::arg("k"), py::arg("radius"))
        .def("predetermined_search_radius", &cube_predetermined_search_radius, py::arg("query"), py::arg("k"), py::arg("radius"))
        .def("fly_search_polygon", &cube_fly_search_polygon, py::arg("query"), py::arg("k"), py::arg("polygon"))
        .def("predetermined_search_polygon", &cube_predetermined_search_polygon, py::arg("query"), py::arg("k"), py::arg("polygon"))
        .def("fly_search_composite", &cube_fly_search_composite, py::arg("query"), py::arg("k"), py::arg("params"))
        .def("predetermined_search_composite", &cube_predetermined_search_composite, py::arg("query"), py::arg("k"), py::arg("params"))
        .def("get_global_bbox", &IndexCube::get_global_bbox, py::return_value_policy::copy)
        .def("get_meta_dim", &IndexCube::get_meta_dim)
        .def("get_metadata", &IndexCube::get_metadata, py::return_value_policy::copy)
        .def("save_index", &IndexCube::save_index, py::arg("path"))
        .def("load_index", &cube_load_index_wrapper, py::arg("path"), py::arg("data_file"));

    py::class_<PostSearchIndex>(m, "PostSearchIndex")
        .def(py::init<size_t, size_t>(),
             py::arg("M") = 16,
             py::arg("ef_construction") = 200)
        .def("load_metadata", &PostSearchIndex::load_metadata, py::arg("metadata_file"))
        .def("build_index", &PostSearchIndex::build_index,
             py::arg("data_file"), py::arg("metadata_file"))
        .def("load_index", &PostSearchIndex::load_index,
             py::arg("index_file"), py::arg("data_file"), py::arg("metadata_file") = "")
        .def("save_index", &PostSearchIndex::save_index, py::arg("index_file"))
        .def("set_ef", &PostSearchIndex::set_ef, py::arg("ef"))
        .def("search", &PostSearchIndex::search,
             py::arg("query"), py::arg("k"), py::arg("bbox"))
        .def("search_with_bounds", &PostSearchIndex::search_with_bounds,
             py::arg("query"), py::arg("k"), py::arg("min_bounds"), py::arg("max_bounds"))
        .def("get_meta_dim", &PostSearchIndex::get_meta_dim)
        .def("get_num_vectors", &PostSearchIndex::get_num_vectors)
        .def("get_metadata", &PostSearchIndex::get_metadata,
             py::return_value_policy::copy);
}
