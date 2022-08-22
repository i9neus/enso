#pragma once

#include <map>
#include "generic/Hash.h"
#include "generic/StdUtils.h"
#include "UIButtonMap.h"

using UIStateID = uint;
using UITriggerHash = uint;
struct UIStateTransition;

enum UIStateTransitionResult : uint
{
    kUIStateOkay = 1,
    kUIStateUpdated = 2,
    kUIStateSucceeded = kUIStateOkay | kUIStateUpdated,

    kUIStateRejected = 4,
    kUIStateError = 0xffffffff
};

enum UIStateTransitionFlags : uint 
{ 
    // These are dummy values. They shouldn't affect the hash so set them to zero    
    kUITriggerOnKeyboard = 0, 
    kUITriggerOnMouseButton = 0,

    kUITriggerOnMouseMove = 1,
    kUITriggerOnMouseWheel = 2,

    kUITriggerAuto = 4
};

struct UIState
{
    enum _attrs : uint { kNull = 0, kInvalid = 0xffffffff };
    
    uint                                              idx;
    std::string                                       id;

    std::function<uint(const uint&, const uint&)>     onExitState;
    std::function<uint(const uint&, const uint&)>     onEnterState;

    uint                                              entryTransitionIdx;
    uint                                              exitTransitionIdx;
};

struct UIStateTransition
{
    enum _attrs : uint { kInvalid = 0xffffffff };

    uint                        hash;

    KeyboardButtonMap           keyTrigger;
    MouseButtonMap              mouseTrigger;
    uint                        triggerFlags;

    uint                        sourceStateIdx;
    uint                        targetStateIdx;

    std::function<std::string(const uint&)> getTargetState;

    inline const bool IsNonDeterministic() const { return getTargetState != nullptr; }
    inline const bool HasDeterministicTarget() const { return targetStateIdx != kInvalid; }
};

class UIStateGraph
{
public:
    UIStateGraph(const KeyboardButtonMap& keyCodes, const MouseButtonMap& mouseCodes) :
        m_keyCodes(keyCodes),
        m_mouseCodes(mouseCodes),
        m_currentState(UIState::kNull)
    {
    }	

    void DeclareState(const std::string& name)
    {
        // Push the new state to the list
        m_uiStateList.push_back(UIState{ uint(m_uiStateList.size()), name, nullptr, nullptr, UIStateTransition::kInvalid, UIStateTransition::kInvalid });
        // Register its index in the map
        m_uiStateMap.emplace(name, m_uiStateList.size() - 1);
    }
    
    template<typename HostType, typename OnEnterState, typename OnLeaveState>
    void DeclareState(const std::string& name, HostType* hostInstance, OnEnterState onEnterState, OnLeaveState onExitState)
    {
        DeclareState(name);        
        m_uiStateList.back().onExitState = std::bind(onExitState, hostInstance, std::placeholders::_1, std::placeholders::_2);
        m_uiStateList.back().onEnterState = std::bind(onEnterState, hostInstance, std::placeholders::_1, std::placeholders::_2);
    }

    template<typename HostType, typename OnEnterState>
    void DeclareState(const std::string& name, HostType* hostInstance, OnEnterState onEnterState)
    {
        DeclareState(name);
        m_uiStateList.back().onEnterState = std::bind(onEnterState, hostInstance, std::placeholders::_1, std::placeholders::_2);
    }

    inline UIState* FindState(const std::string& name)
    { 
        auto it = m_uiStateMap.find(name);
        return (it == m_uiStateMap.end()) ? nullptr : &m_uiStateList[it->second];
    }

    inline const UIState& GetState(const uint idx) const 
    {
        AssertMsg(idx < m_uiStateList.size(), "Invalid state index.");
        return m_uiStateList[idx];
    }

    inline std::string GetStateID(const uint idx) const
    {
        return GetState(idx).id;
    }

    inline void SetState(const std::string& id)
    {
        auto it = m_uiStateMap.find(id);
        AssertMsgFmt(it != m_uiStateMap.end(), "Could not set state to '%s'. State not found.", id.c_str());
        m_currentState = it->second;
    }

    inline void Reset()
    {
        m_currentState = 0;
    }

    void DeclareTransitionImpl(const std::string& sourceStateID, const std::string& targetStateID, const KeyboardButtonMap& keyTrigger, 
                               const MouseButtonMap& mouseTrigger, const uint& triggerFlags, const bool isAuto)
    {
        UIState* sourceState = FindState(sourceStateID);
        AssertMsgFmt(sourceState, "Error: transition sourceState '%s' was not declared.", sourceStateID.c_str());

        // Each state can only have one exit transition object 
        //AssertMsgFmt(sourceState->exitTransitionIdx == UIStateTransition::kInvalid, "Error: cannot bind transition {'%s' -> '%s'} because state '%s' already has a transition bound to it",
        //    sourceStateID.c_str(), targetStateID.c_str(), sourceStateID.c_str());

        UIState* targetState = nullptr;
        if (!targetStateID.empty())
        {
            targetState = FindState(targetStateID);
            AssertMsgFmt(targetState, "Error: transition targetState '%s' was not declared.", targetStateID.c_str());
        }        
        
        // Generate a hash from the origin state's ID and the trigger required to transition away from it
        const uint hash = HashCombine(HashOf(sourceState->idx), keyTrigger.HashOf(), mouseTrigger.HashOf(), HashOf(triggerFlags));

        //Log::Error("Hash: 0x%x, 0x%x, 0x%x -> 0x%x", uint(std::hash<uint>{}(0)), keyTrigger.HashOf(), mouseTrigger.HashOf(), hash);

        m_uiTransitionList.push_back(UIStateTransition{ hash,
                                             keyTrigger,
                                             mouseTrigger,
                                             triggerFlags,
                                             sourceState->idx, 
                                             targetState ? targetState->idx : UIState::kInvalid,
                                             nullptr });

        // Mark the source and target states and being connected up 
        sourceState->exitTransitionIdx = m_uiTransitionList.size() - 1;
        if (targetState)
        {
            targetState->entryTransitionIdx = m_uiTransitionList.size() - 1;
        }

        // Use either the hash or the index as key depending on the container type
        if (!isAuto)
        {
            m_uiStateTriggeredTransitionMap.emplace(hash, m_uiTransitionList.size() - 1);
        }
        else
        {
            AssertMsgFmt(!Contains(m_uiStateAutoTransitionMap, sourceState->idx),
                         "Error: an automatic transition from state '%s' has already been declared.", sourceStateID.c_str());
            m_uiStateAutoTransitionMap.emplace(sourceState->idx, m_uiTransitionList.size() - 1);
        }
    }

    inline void DeclareDeterministicTransition(const std::string& sourceStateID, const std::string& targetStateID, const KeyboardButtonMap& keyTrigger, 
                                               const MouseButtonMap& mouseTrigger, const uint& triggerFlags)
    {
        DeclareTransitionImpl(sourceStateID, targetStateID, keyTrigger, mouseTrigger, triggerFlags, false);
    }

    template<class HostClass, typename GetTransitionState>
    void DeclareNonDeterministicTransition(const std::string& sourceState, const KeyboardButtonMap& keyTrigger, const MouseButtonMap& mouseTrigger, const uint& triggerFlags,
                                           HostClass* hostClass, GetTransitionState getTargetState)
    {
        DeclareTransitionImpl(sourceState, "", keyTrigger, mouseTrigger, triggerFlags, false);

        m_uiTransitionList.back().getTargetState = std::bind(getTargetState, hostClass, std::placeholders::_1);
    }

    inline void DeclareAutoTransition(const std::string& sourceStateID, const std::string& targetStateID)
    {
        DeclareTransitionImpl(sourceStateID, targetStateID, nullptr, nullptr, 0, true);
    }

    bool MakeTransition(const UIStateTransition& transition)
    {
        // If the source state has a notification lambda, call it here
        uint sourceStateIdx = m_currentState;
        if (m_uiStateList[m_currentState].onExitState)
        {
            // Forbid allowing exit functors to change the graph state (for now)
            AssertMsgFmt(m_currentState == sourceStateIdx, "Illegal opereation: exit functor '%s' manually changed the graph state.", m_uiStateList[m_currentState].id.c_str());
            
            const uint result = m_uiStateList[m_currentState].onExitState(sourceStateIdx, UIState::kInvalid);
            AssertMsg(result != kUIStateError, "State transition error.");
            if (result == kUIStateRejected)
            {
                Log::Warning("Warning: failed to enter state '%s': transition was rejected.", m_uiStateList[m_currentState].id);
                return true;
            }
        }

        // If the transition lambda attached then it's non-deterministic. Call the lambda to determine what state we're migrating to
        if (transition.IsNonDeterministic())
        {
            const std::string newStateID = transition.getTargetState(sourceStateIdx);
            AssertMsgFmt(!newStateID.empty(),
                "Error: non-deterministic transition from '%s' failed: state is empty.", m_uiStateList[m_currentState].id.c_str());

            const UIState* newState = FindState(newStateID);
            AssertMsgFmt(newState, "Error: non-deterministic transition from '%s' failed: '%s' is not a valid state.", m_uiStateList[m_currentState].id.c_str(), newStateID.c_str());

            m_currentState = newState->idx;
        }
        // Otherwise, treat the transition as deterministic
        else
        {
            m_currentState = transition.targetStateIdx;
        }

        // If the target (now the current) state has a lambda, call it now.
        if (m_uiStateList[m_currentState].onEnterState)
        {
            // Allow the target state entry functor to change the state. Keep cycling until 
            uint targetState = UIState::kInvalid;
            constexpr int kMaxTransitions = 10;
            int numTransitions = 0;
            do
            {
                targetState = m_currentState;
                const uint result = m_uiStateList[m_currentState].onEnterState(sourceStateIdx, m_currentState);
                AssertMsg(result != kUIStateError, "State transition error.");

                if (result == kUIStateRejected)
                {
                    Log::Warning("Warning: failed to enter state '%s': transition was rejected.",
                        transition.IsNonDeterministic() ? std::string("[UNKNOWN]") : m_uiStateList[transition.targetStateIdx].id);
                    return true;
                }
            }
            while (m_currentState != targetState && ++numTransitions < kMaxTransitions);

            AssertMsg(numTransitions < kMaxTransitions, "State exceeded maximum allowed number of transitions. Possible cycle detected.");
        }

        Log::Success("State changed: %s -> %s", m_uiStateList[sourceStateIdx].id, m_uiStateList[m_currentState].id);
        return true;
    }

	void OnTriggerTransition(const uint triggerFlags)
	{
        uint hash = HashCombine(HashOf(m_currentState), m_keyCodes.HashOf(), m_mouseCodes.HashOf(), HashOf(triggerFlags));

        // Search for states that match the current set of triggers
		auto range = m_uiStateTriggeredTransitionMap.equal_range(hash);
		for (auto it = range.first; it != range.second; ++it)
		{
            // If this transition doesn't match the trigger criteria, continue looking
            const auto& transition = m_uiTransitionList[it->second];
            if (transition.sourceStateIdx != m_currentState || transition.keyTrigger != m_keyCodes || 
                transition.mouseTrigger != m_mouseCodes || transition.triggerFlags != triggerFlags) { continue; }
	
            MakeTransition(transition);
		}

        // Iteratively execute any auto-transitions that exist for the current
        for(auto it = m_uiStateAutoTransitionMap.find(m_currentState); 
            it != m_uiStateAutoTransitionMap.end(); 
            it = m_uiStateAutoTransitionMap.find(m_currentState))
        {
            MakeTransition(m_uiTransitionList[it->second]);
        }
	}

    void Finalise() const
    {
        std::vector<const UIState*> acyclicStates;
        std::vector<const UIState*> orphanedStates;
        for (const auto& state : m_uiStateList)
        {
            if (state.exitTransitionIdx == UIState::kInvalid && state.entryTransitionIdx == UIState::kInvalid) { orphanedStates.push_back(&state); }
            else if (state.exitTransitionIdx == UIState::kInvalid) { acyclicStates.push_back(&state); }
        }
        
        Log::Indent indent("Built UI graph:");
        Log::Write("Start state: %s", m_uiStateList[m_currentState].id);
        Log::Write("%i nodes", m_uiStateList.size());
        Log::Write("%i edges", m_uiTransitionList.size());       
        
        if (!acyclicStates.empty())
        {
            Log::Warning("%i acyclic transitions detected! The following nodes are leaves.", acyclicStates.size());
            for (const auto state : acyclicStates) { Log::Warning("  - %s", state->id); }
        }
        else
        {
            Log::Success("No acyclic walks!");
        }

        if (!orphanedStates.empty())
        {
            Log::Warning("%i orphaned nodes detected! The following have no transitions in or out.", orphanedStates.size());
            for (const auto state : orphanedStates) { Log::Warning("  - %s", state->id); }
        }
        else
        {
            Log::Success("No orphaned states!");
        }
    }

private:
    std::vector<UIState>                            m_uiStateList;
    std::map<std::string, uint>                     m_uiStateMap;
    std::vector<UIStateTransition>                  m_uiTransitionList;
    std::unordered_multimap<uint, uint>             m_uiStateTriggeredTransitionMap;
    std::unordered_map<uint, uint>                  m_uiStateAutoTransitionMap;

	const KeyboardButtonMap&                        m_keyCodes;
	const MouseButtonMap&   						m_mouseCodes;

    uint                                            m_currentState;
};