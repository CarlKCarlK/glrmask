//! Recognize bounded-repeat shapes eligible for exact virtual state representations.

use crate::automata::lexer::ast::Expr;
use crate::automata::lexer::dfa::DFA;
use crate::automata::lexer::runtime_repeat_product::{
    VirtualBinaryRepeatIntersectionDescriptor,
    VirtualBoundedRepeatSpec,
};
use crate::automata::lexer::runtime_unit_repeat::virtual_unit_repeat_state_ids_fit;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::ds::u8set::U8Set;
use std::sync::Arc;
use super::bounded_repeat::cached_direct_bounded_repeat_base_dfa_unconditionally;
use super::deferred::expression_contains_large_bounded_repeat;
use super::factor::unwrap_shared;
use super::nfa::expr_u8set;

/// Return the exact byte language of an expression which consumes precisely
/// one byte.  Keeping this proof deliberately syntactic makes the aligned
/// repeat-intersection rewrite fail closed for variable-width bodies.
pub(super) fn exact_unit_byte_language(expr: &Expr) -> Option<U8Set> {
    match unwrap_shared(expr) {
        Expr::U8Seq(bytes) if bytes.len() == 1 => Some(U8Set::single(bytes[0])),
        Expr::U8Class(bytes) => Some(*bytes),
        Expr::Choice(options) => {
            let mut bytes = U8Set::empty();
            for option in options {
                bytes |= exact_unit_byte_language(option)?;
            }
            Some(bytes)
        }
        Expr::Intersect { expr, intersect } => Some(
            exact_unit_byte_language(expr)?.intersection(&exact_unit_byte_language(intersect)?),
        ),
        _ => None,
    }
}

#[doc(hidden)]
pub fn virtual_zero_min_unit_repeat_fits_state_ids(
    max: usize,
    physical_state_count: u32,
) -> bool {
    virtual_unit_repeat_state_ids_fit(max, physical_state_count)
}

/// Recognize the exact arithmetic runtime lane after factoring. The body proof
/// is syntactic and therefore fails closed for any variable-width repetition.
#[doc(hidden)]
pub fn virtual_unit_repeat_descriptor(expr: &Expr) -> Option<(U8Set, usize, usize)> {
    let Expr::Repeat {
        expr: body,
        min,
        max: Some(max),
    } = unwrap_shared(expr)
    else {
        return None;
    };
    // The standalone arithmetic runtime has one physical reset state and
    // reserves the high raw-state bit. Keep descriptor eligibility identical
    // to that state-ID contract. Larger exact repeats can still use the
    // binary repeat-product lane, where the repetition count is not encoded as
    // a contiguous virtual-state interval.
    if !virtual_zero_min_unit_repeat_fits_state_ids(*max, 1) {
        return None;
    }
    let bytes = exact_unit_byte_language(body)?;
    (!bytes.is_empty() && *max > 0 && min <= max).then_some((bytes, *min, *max))
}

#[doc(hidden)]
pub fn virtual_zero_min_unit_repeat_descriptor(expr: &Expr) -> Option<(U8Set, usize)> {
    let (body, min, max) = virtual_unit_repeat_descriptor(expr)?;
    (min == 0).then_some((body, max))
}

/// Build the O(1)-storage physical tokenizer for the supported standalone
/// repeat. Positive-length residuals are supplied by `Tokenizer`'s arithmetic
/// virtual-state sidecar.
pub fn build_virtual_unit_repeat_tokenizer(expressions: &[Expr]) -> Option<Tokenizer> {
    let [expression] = expressions else {
        return None;
    };
    let (body, min, max) = virtual_unit_repeat_descriptor(expression)?;
    let mut tokenizer = Tokenizer::from_parts(
        DFA::new(1),
        1,
        Some(Arc::from(expressions.to_vec().into_boxed_slice())),
    );
    tokenizer.install_virtual_unit_repeat(body, min, max)?;
    Some(tokenizer)
}

pub fn build_virtual_zero_min_unit_repeat_tokenizer(expressions: &[Expr]) -> Option<Tokenizer> {
    let [expression] = expressions else {
        return None;
    };
    virtual_zero_min_unit_repeat_descriptor(expression)?;
    build_virtual_unit_repeat_tokenizer(expressions)
}

pub(super) const VIRTUAL_BINARY_REPEAT_MIN_BOUND: usize = 4_096;

fn virtual_bounded_repeat_spec(expr: &Expr) -> Option<VirtualBoundedRepeatSpec> {
    let Expr::Repeat {
        expr: body,
        min,
        max: Some(max),
    } = unwrap_shared(expr)
    else {
        return None;
    };
    if *max < VIRTUAL_BINARY_REPEAT_MIN_BOUND || min > max {
        return None;
    }
    // The body is compiled below in order to prove the deterministic,
    // non-nullable, prefix-free residual model. Do not perform that semantic
    // probe if the body itself contains another giant bounded repeat: doing so
    // would eagerly materialize exactly the state space this virtual lane is
    // intended to avoid.
    if expression_contains_large_bounded_repeat(body) {
        return None;
    }
    let base_dfa = cached_direct_bounded_repeat_base_dfa_unconditionally(body, None)?;
    Some(VirtualBoundedRepeatSpec {
        base_dfa,
        min: u32::try_from(*min).ok()?,
        max: u32::try_from(*max).ok()?,
    })
}

fn virtual_zero_min_bounded_repeat_spec(expr: &Expr) -> Option<VirtualBoundedRepeatSpec> {
    let spec = virtual_bounded_repeat_spec(expr)?;
    (spec.min == 0).then_some(spec)
}

/// Recognize an exact pure intersection whose two large bounded-repeat
/// coordinates can remain symbolic at runtime. Each body must be deterministic,
/// non-nullable and prefix-free; that makes `(completed copies, body state)` a
/// complete residual coordinate for each side.
#[doc(hidden)]
pub fn virtual_binary_bounded_repeat_intersection_descriptor(
    expr: &Expr,
) -> Option<VirtualBinaryRepeatIntersectionDescriptor> {
    let Expr::Intersect { expr, intersect } = unwrap_shared(expr) else {
        return None;
    };
    let left = virtual_zero_min_bounded_repeat_spec(expr)?;
    let right = virtual_zero_min_bounded_repeat_spec(intersect)?;
    let byte_support = expr_u8set(expr).intersection(&expr_u8set(intersect));
    (!byte_support.is_empty()).then_some(VirtualBinaryRepeatIntersectionDescriptor {
        left,
        right,
        byte_support,
    })
}

/// Recognize one large bounded repetition whose body admits the
/// exact symbolic repeat coordinate used by the lazy product runtime.
///
/// The runtime already implements the exact language intersection of two
/// bounded-repeat coordinates.  Supplying the same coordinate on both sides
/// therefore represents the original language exactly by `L = L ∩ L`, while
/// avoiding `(max + 1) * body_states` materialization.
#[doc(hidden)]
pub fn virtual_large_bounded_repeat_descriptor(
    expr: &Expr,
) -> Option<VirtualBinaryRepeatIntersectionDescriptor> {
    let spec = virtual_bounded_repeat_spec(expr)?;
    let byte_support = expr_u8set(expr);
    (!byte_support.is_empty()).then_some(VirtualBinaryRepeatIntersectionDescriptor {
        left: spec.clone(),
        right: spec,
        byte_support,
    })
}

/// Return the declared upper bound of a top-level bounded repetition large
/// enough to require symbolic treatment. This deliberately ignores whether
/// the body is currently supported by the exact symbolic backend: callers use
/// it to fail closed instead of falling through to the eager repeat compiler.
#[doc(hidden)]
pub fn large_top_level_bounded_repeat_bound(expr: &Expr) -> Option<usize> {
    let Expr::Repeat { max: Some(max), .. } = unwrap_shared(expr) else {
        return None;
    };
    (*max >= VIRTUAL_BINARY_REPEAT_MIN_BOUND).then_some(*max)
}
