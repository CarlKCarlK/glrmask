//! Borrowed runtime weights over materialized, packed-DWA, and packed-pool storage.

use crate::automata::weighted::dwa::PackedRuntimeTokenSetRef;
use crate::automata::weighted::dwa::PackedRuntimeWeightRef;
use crate::ds::weight::PackedRuntimePoolTokenSetRef;
use crate::ds::weight::PackedRuntimePoolWeightRef;
use crate::ds::weight::Weight;
use range_set_blaze::RangeSetBlaze;
use std::sync::Arc;

#[derive(Clone, Copy)]
pub(crate) enum RuntimeTokenSetRef<'a> {
    Materialized(&'a Arc<RangeSetBlaze<u32>>),
    PackedDwa(PackedRuntimeTokenSetRef<'a>),
    PackedPool(PackedRuntimePoolTokenSetRef<'a>),
}


impl<'a> RuntimeTokenSetRef<'a> {
    #[inline]
    pub(crate) fn is_empty(self) -> bool {
        match self {
            Self::Materialized(tokens) => tokens.is_empty(),
            Self::PackedDwa(tokens) => tokens.range_count() == 0,
            Self::PackedPool(tokens) => tokens.is_empty(),
        }
    }

    #[inline]
    pub(crate) fn materialized_key(self) -> Option<usize> {
        match self {
            Self::Materialized(tokens) => Some(Arc::as_ptr(tokens) as usize),
            Self::PackedDwa(_) | Self::PackedPool(_) => None,
        }
    }

    #[inline]
    pub(crate) fn packed_id(self) -> Option<u32> {
        match self {
            Self::Materialized(_) => None,
            Self::PackedDwa(tokens) => Some(tokens.id()),
            Self::PackedPool(_) => None,
        }
    }

    #[inline]
    pub(crate) fn packed_pool_id(self) -> Option<u32> {
        match self {
            Self::PackedPool(tokens) => Some(tokens.id()),
            _ => None,
        }
    }

    #[inline]
    pub(crate) fn for_each_range(self, mut f: impl FnMut(u32, u32)) {
        match self {
            Self::Materialized(tokens) => {
                for range in tokens.ranges() {
                    f(*range.start(), *range.end());
                }
            }
            Self::PackedDwa(tokens) => {
                tokens.for_each_range(f);
            }
            Self::PackedPool(tokens) => {
                tokens.for_each_range(f);
            }
        }
    }

    #[inline]
    pub(crate) fn word_spans(self) -> Option<u32> {
        match self {
            Self::Materialized(_) => None,
            Self::PackedDwa(tokens) => Some(tokens.word_spans()),
            Self::PackedPool(_) => None,
        }
    }

    pub(crate) fn to_range_set(self) -> RangeSetBlaze<u32> {
        match self {
            Self::Materialized(tokens) => tokens.as_ref().clone(),
            packed => {
                let mut ranges = Vec::new();
                packed.for_each_range(|start, end| ranges.push(start..=end));
                RangeSetBlaze::from_iter(ranges)
            }
        }
    }
}


#[derive(Clone, Copy)]
pub(crate) enum RuntimeWeightRef<'a> {
    Materialized(&'a Weight),
    PackedDwa(PackedRuntimeWeightRef<'a>),
    PackedPool(PackedRuntimePoolWeightRef<'a>),
}


impl<'a> RuntimeWeightRef<'a> {
    #[inline]
    pub(crate) fn is_full(self) -> bool {
        match self {
            Self::Materialized(weight) => weight.is_full(),
            Self::PackedDwa(weight) => weight.is_full(),
            Self::PackedPool(weight) => weight.is_full(),
        }
    }

    #[inline]
    pub(crate) fn is_empty(self) -> bool {
        match self {
            Self::Materialized(weight) => weight.is_empty(),
            Self::PackedDwa(weight) => weight.is_empty(),
            Self::PackedPool(weight) => weight.is_empty(),
        }
    }

    #[inline]
    pub(crate) fn token_set_for_tsid(self, tsid: u32) -> Option<RuntimeTokenSetRef<'a>> {
        match self {
            Self::Materialized(weight) => weight
                .token_set_for_tsid_ref(tsid)
                .map(RuntimeTokenSetRef::Materialized),
            Self::PackedDwa(weight) => weight.token_set_for_tsid(tsid).map(|tokens| {
                tokens.materialized_arc().map_or(
                    RuntimeTokenSetRef::PackedDwa(tokens),
                    RuntimeTokenSetRef::Materialized,
                )
            }),
            Self::PackedPool(weight) => weight
                .token_set_for_tsid(tsid)
                .map(RuntimeTokenSetRef::PackedPool),
        }
    }

    #[inline]
    pub(crate) fn to_weight(self) -> Weight {
        if self.is_full() {
            return Weight::all();
        }
        if self.is_empty() {
            return Weight::empty();
        }
        let mut entries = Vec::<(u32, RangeSetBlaze<u32>)>::new();
        self.for_each_entry(|start, end, tokens| {
            let tokens = tokens.to_range_set();
            entries.extend((start..=end).map(|tsid| (tsid, tokens.clone())));
        });
        Weight::from_per_tsid_token_sets(entries)
    }

    pub(crate) fn for_each_entry(
        self,
        mut f: impl FnMut(u32, u32, RuntimeTokenSetRef<'a>),
    ) {
        match self {
            Self::Materialized(weight) => {
                if weight.is_full() {
                    return;
                }
                for (range, tokens) in weight.raw_range_values() {
                    f(
                        *range.start(),
                        *range.end(),
                        RuntimeTokenSetRef::Materialized(tokens),
                    );
                }
            }
            Self::PackedDwa(weight) => {
                for ((start, end), tokens) in weight.entries() {
                    let tokens = tokens.materialized_arc().map_or(
                        RuntimeTokenSetRef::PackedDwa(tokens),
                        RuntimeTokenSetRef::Materialized,
                    );
                    f(start, end, tokens);
                }
            }
            Self::PackedPool(weight) => {
                for ((start, end), tokens) in weight.entries() {
                    f(start, end, RuntimeTokenSetRef::PackedPool(tokens));
                }
            }
        }
    }
}


impl<'a> From<&'a Weight> for RuntimeWeightRef<'a> {
    #[inline]
    fn from(weight: &'a Weight) -> Self {
        Self::Materialized(weight)
    }
}
