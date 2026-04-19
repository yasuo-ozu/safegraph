use safegraph::graph::edge::{Endpoints, Map};
use std::collections::{BTreeSet, HashSet};

#[test]
fn array_endpoints_iter() {
    let eps: [u32; 2] = [1, 2];
    let items: Vec<u32> = eps.into_iter().collect();
    assert_eq!(items, vec![1, 2]);
}

#[test]
fn array_endpoints_clone_iter() {
    let eps: [u32; 2] = [10, 20];
    let items: Vec<u32> = eps.iter().collect();
    assert_eq!(items, vec![10, 20]);
}

#[test]
fn array_endpoints_eq() {
    let a: [u32; 2] = [1, 2];
    let b: [u32; 2] = [1, 2];
    let c: [u32; 2] = [2, 1];
    assert_eq!(a, b);
    assert_ne!(a, c);
}

#[test]
fn array_endpoints_map_forward() {
    let eps: [u32; 2] = [1, 2];
    let mapped: [u64; 2] = eps.map_forward(|x| x as u64);
    assert_eq!(mapped, [1u64, 2u64]);
}

#[test]
fn array_endpoints_map_backward() {
    let mapped: [u64; 2] = [10, 20];
    let original: [u32; 2] = <[u32; 2]>::map_backward(mapped, |x| x as u32);
    assert_eq!(original, [10, 20]);
}

#[test]
fn array_endpoints_try_from_sources_targets() {
    let eps = <[u32; 2]>::try_from_sources_targets([5], [10]).unwrap();
    assert_eq!(eps, [5, 10]);
}

#[test]
fn hashset_endpoints_iter() {
    let eps: HashSet<u32> = [1, 2, 3].into_iter().collect();
    assert_eq!(eps.len(), 3);
}

#[test]
fn hashset_endpoints_map() {
    let eps: HashSet<u32> = [1, 2].into_iter().collect();
    let mapped: HashSet<u64> = eps.map_forward(|x| x as u64);
    assert!(mapped.contains(&1u64));
    assert!(mapped.contains(&2u64));
}

#[test]
fn hashset_endpoints_map_backward() {
    let mapped: HashSet<u64> = [10, 20].into_iter().collect();
    let original: HashSet<u32> = <HashSet<u32>>::map_backward(mapped, |x| x as u32);
    assert!(original.contains(&10));
    assert!(original.contains(&20));
}

#[test]
fn btreeset_endpoints_iter() {
    let eps: BTreeSet<u32> = [3, 1, 2].into_iter().collect();
    let items: Vec<u32> = eps.into_iter().collect();
    assert_eq!(items, vec![1, 2, 3]); // sorted
}

#[test]
fn btreeset_endpoints_map() {
    let eps: BTreeSet<u32> = [1, 2].into_iter().collect();
    let mapped: BTreeSet<u64> = eps.map_forward(|x| x as u64 * 10);
    let items: Vec<u64> = mapped.into_iter().collect();
    assert_eq!(items, vec![10, 20]);
}

#[test]
fn btreeset_endpoints_map_backward() {
    let mapped: BTreeSet<u64> = [10, 20].into_iter().collect();
    let original: BTreeSet<u32> = <BTreeSet<u32>>::map_backward(mapped, |x| x as u32);
    let items: Vec<u32> = original.into_iter().collect();
    assert_eq!(items, vec![10, 20]);
}
