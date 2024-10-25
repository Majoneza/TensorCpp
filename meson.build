project('TensorCpp', 'cpp',
    version: '1.0.0', license: 'MIT')

subdir('include')

pkg_mod = import('pkgconfig')
pkg_mod.generate(
    name: meson.project_name().to_lower(),
    version: meson.project_version(),
    description: 'Tensor C++ library')
